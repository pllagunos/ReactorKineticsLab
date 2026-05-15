"""2D r-z time-dependent one-group diffusion solver with 6 delayed-neutron groups.

The time-dependent equation uses a k_ref scaling factor so that the initial
critical state is exactly stationary on the discrete mesh:

    1/v * d_phi/dt = -L phi + (1-beta) * (F/k_ref) * phi + sum_i lambda_i * C_i
    d C_i/dt = beta_i * (nuSigma_f / k_ref) * phi - lambda_i * C_i

where:
  L        = loss matrix (absorption + leakage), depends on rod position
  F/k_ref  = scaled fission matrix; k_ref is the eigenvalue at the reference state
  C_i      = spatial precursor concentration field (Nr*Nz vector) for group i
  v        = one-group thermal neutron velocity (cm/s)

Integration scheme: fully implicit Euler (unconditionally stable).
LU factorization of the 2D operator matrix is cached and reused as long as
the rod insertion fraction and time step remain unchanged.
"""

from __future__ import annotations

import logging
import time as _time_module
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import DELAYED_NEUTRON_GROUPS
from .diffusion import Model2D, build_2d_matrices, solve_2d

logger = logging.getLogger(__name__)

# One-group thermal neutron velocity in D2O (cm/s).
# At 0.025 eV (2200 m/s), this is the standard thermal-average value.
NEUTRON_VELOCITY_CM_S: float = 2.2e5

# Mesh spacing for the transient solver.
# 5 cm is coarser than the core_service 3 cm mesh; this keeps the LU
# factorization fast enough for real-time interactive stepping.
TRANSIENT_DR_CM: float = 5.0
TRANSIENT_DZ_CM: float = 5.0

# Fixed time step for implicit Euler (simulation seconds).
TRANSIENT_DT_S: float = 1.0

# Display resolution — downsampled from solver mesh for heatmap rendering
TRANSIENT_NR_DISPLAY: int = 40
TRANSIENT_NZ_DISPLAY: int = 60

BETA_TOTAL: float = sum(g.beta for g in DELAYED_NEUTRON_GROUPS)


@dataclass
class TransientSolverState:
    """Mutable state of the transient solver (absolute amplitude, not renormalized)."""

    phi: np.ndarray           # (nr*nz,) — amplitude grows/decays naturally
    precursors: list[np.ndarray]   # 6 × (nr*nz,)
    phi_ref_power: float      # Σ(nuSigma_f * phi_0 * V) at t=0, for P_norm
    time_s: float
    step_count: int


class TransientSolver:
    """Implicit Euler 2D r-z transient diffusion stepper.

    Usage
    -----
    solver = TransientSolver(model, dr, dz, dt, v)
    state, k_ref = solver.initialize(rod_frac)  # initial critical state
    state = solver.step(state, rod_frac)        # advance one dt
    phi2d, meta = solver.display_data(state)   # downsampled heatmap for UI
    """

    def __init__(
        self,
        model: Model2D,
        dr: float = TRANSIENT_DR_CM,
        dz: float = TRANSIENT_DZ_CM,
        dt: float = TRANSIENT_DT_S,
        v: float = NEUTRON_VELOCITY_CM_S,
    ) -> None:
        self._model = model
        self.dr = dr
        self.dz = dz
        self.dt = dt
        self.v = v

        self._nr = int(round(model.R_extrap / dr))
        self._nz = int(round(model.H_extrap / dz))
        self._cell_count = self._nr * self._nz

        # Cell-centre coordinates
        self._r_grid_full = (np.arange(self._nr) + 0.5) * dr
        self._z_grid_full = -model.H_extrap / 2 + (np.arange(self._nz) + 0.5) * dz

        # Cylindrical volume weights per cell: r_i * dr * dz (2π cancels in ratios).
        # Flat array is row-major (r is slow index, z is fast): repeat each r value nz times.
        self._vol_weights = np.repeat(self._r_grid_full * dr * dz, self._nz)  # (cell_count,)

        # Cache for the LU factorization
        self._cached_rod_frac: float | None = None
        self._cached_k_ref: float | None = None
        self._lu = None
        self._f_scaled: np.ndarray | None = None  # nuSigma_f / k_ref, per cell

    # ------------------------------------------------------------------
    # Internal matrix assembly
    # ------------------------------------------------------------------

    def _rebuild(self, rod_frac: float, k_ref: float) -> None:
        """Rebuild and factorize the implicit Euler matrix A."""
        loss, fission, _r, _z, nr, nz = build_2d_matrices(
            self._model, self.dr, self.dz, rod_frac
        )
        assert nr == self._nr and nz == self._nz

        f_raw = fission.diagonal()           # nuSigma_f per cell
        f_scaled = f_raw / k_ref             # scaled fission source (1/cm)

        # Effective fission coefficient in the implicit flux equation:
        #   b_eff = (1-β) + Σ_i β_i λ_i dt / (1 + λ_i dt)
        dt = self.dt
        delayed_coeff = sum(
            g.beta * g.decay_constant * dt / (1.0 + g.decay_constant * dt)
            for g in DELAYED_NEUTRON_GROUPS
        )
        b_eff = (1.0 - BETA_TOTAL) + delayed_coeff  # dimensionless (not yet /k_ref)

        # A = L + (1/v/dt) I - (b_eff / k_ref) * diag(nuSigma_f)
        #   = loss + diag(1/v/dt - b_eff * f_scaled)
        inv_v_dt = 1.0 / (self.v * dt)
        diag_extra = inv_v_dt - b_eff * f_scaled

        A = loss + sp.diags(diag_extra, format="csr")

        t0 = _time_module.perf_counter()
        self._lu = spla.factorized(A.tocsc())
        logger.info(
            "Transient LU factorized in %.3f s  (rod_frac=%.4f, nr=%d, nz=%d, cells=%d)",
            _time_module.perf_counter() - t0,
            rod_frac,
            nr,
            nz,
            self._cell_count,
        )

        self._cached_rod_frac = rod_frac
        self._cached_k_ref = k_ref
        self._f_scaled = f_scaled

    def _ensure_factorized(self, rod_frac: float, k_ref: float) -> None:
        rod_frac_r = round(rod_frac, 4)  # 4 dp ≈ 0.01 % resolution
        if (
            self._lu is not None
            and self._cached_rod_frac == rod_frac_r
            and self._cached_k_ref == k_ref
        ):
            return
        self._rebuild(rod_frac_r, k_ref)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, rod_frac: float) -> tuple[TransientSolverState, float]:
        """Compute the steady-state initial condition via an eigenvalue solve.

        Returns
        -------
        state : TransientSolverState
        k_ref : float
            k_eff from the eigenvalue solve; used as the criticality reference.
        """
        t0 = _time_module.perf_counter()
        sol = solve_2d(self._model, dr=self.dr, dz=self.dz, x_insert=rod_frac)
        k_ref = float(sol["k_eff"])
        logger.info(
            "Transient init: solve_2d done in %.2f s  (rod_frac=%.4f, k_eff=%.6f)",
            _time_module.perf_counter() - t0,
            rod_frac,
            k_ref,
        )

        self._ensure_factorized(rod_frac, k_ref)

        # Initial flux: eigenvalue solution as a flat array, peak-normalized to 1.
        phi_0 = sol["phi"].ravel().copy()  # already peak-normalized by solve_2d

        # Steady-state precursors: C_i,0 = β_i (nuSigma_f/k_ref) φ_0 / λ_i
        f_scaled = self._f_scaled
        precursors_0 = [
            g.beta * f_scaled * phi_0 / g.decay_constant
            for g in DELAYED_NEUTRON_GROUPS
        ]

        # Reference power integral (cylindrical volume-weighted fission rate)
        # Store so we can compute P_norm later without needing phi_0 again.
        phi_ref_power = float(
            np.dot(f_scaled * k_ref * self._vol_weights, phi_0)
        )  # unnormalized; P_norm(t) = dot(..., phi(t)) / phi_ref_power

        state = TransientSolverState(
            phi=phi_0,
            precursors=precursors_0,
            phi_ref_power=phi_ref_power,
            time_s=0.0,
            step_count=0,
        )
        return state, k_ref

    def step(
        self,
        state: TransientSolverState,
        rod_frac: float,
        k_ref: float,
    ) -> TransientSolverState:
        """Advance one implicit Euler time step.

        The flux amplitude grows/decays naturally — no internal renormalization.
        """
        self._ensure_factorized(round(rod_frac, 4), k_ref)

        phi_n = state.phi
        dt = self.dt
        inv_v_dt = 1.0 / (self.v * dt)

        # RHS: b = (1/v/dt) φ^n + Σ_i [λ_i / (1 + λ_i dt)] C_i^n
        b = inv_v_dt * phi_n
        for i, g in enumerate(DELAYED_NEUTRON_GROUPS):
            eps_i = g.decay_constant / (1.0 + g.decay_constant * dt)
            b = b + eps_i * state.precursors[i]

        # Solve: A φ^{n+1} = b
        phi_n1 = self._lu(b)
        phi_n1 = np.maximum(phi_n1, 0.0)

        # Update precursors: C_i^{n+1} = (C_i^n + dt β_i f_scaled φ^{n+1}) / (1 + λ_i dt)
        f_scaled = self._f_scaled
        new_precursors = [
            (state.precursors[i] + dt * g.beta * f_scaled * phi_n1)
            / (1.0 + g.decay_constant * dt)
            for i, g in enumerate(DELAYED_NEUTRON_GROUPS)
        ]

        return TransientSolverState(
            phi=phi_n1,
            precursors=new_precursors,
            phi_ref_power=state.phi_ref_power,
            time_s=state.time_s + dt,
            step_count=state.step_count + 1,
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def power_norm(self, state: TransientSolverState) -> float:
        """P/P_0: cylindrical volume-weighted fission rate ratio.

        Uses the stored k_ref (absorbed into f_scaled * k_ref = nuSigma_f).
        """
        if state.phi_ref_power <= 0:
            return 1.0
        f_raw = self._f_scaled * self._cached_k_ref
        current = float(np.dot(f_raw * self._vol_weights, state.phi))
        return current / state.phi_ref_power

    def display_phi(
        self,
        state: TransientSolverState,
        nr_d: int = TRANSIENT_NR_DISPLAY,
        nz_d: int = TRANSIENT_NZ_DISPLAY,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return downsampled heatmap, r_grid, z_grid for UI display.

        heatmap is peak-normalized (display copy, not the internal state).
        """
        phi2d = state.phi.reshape(self._nr, self._nz)
        peak = phi2d.max() or 1.0
        phi_norm = phi2d / peak

        # Downsample
        r_idx = _downsample_idx(self._nr, nr_d)
        z_idx = _downsample_idx(self._nz, nz_d)
        phi_d = phi_norm[np.ix_(r_idx, z_idx)]
        r_d = self._r_grid_full[r_idx]
        z_d = self._z_grid_full[z_idx]
        return phi_d, r_d, z_d

    def radial_profile(
        self, state: TransientSolverState
    ) -> tuple[np.ndarray, np.ndarray]:
        """Midplane (z≈0) radial flux profile, normalized to local peak."""
        phi2d = state.phi.reshape(self._nr, self._nz)
        iz_mid = int(np.argmin(np.abs(self._z_grid_full)))
        radial = phi2d[:, iz_mid]
        peak = radial.max() or 1.0
        return self._r_grid_full.copy(), radial / peak

    def axial_profile(
        self, state: TransientSolverState, r_fuel_mid_cm: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Axial flux profile at the fuel annulus mid-radius, normalized to local peak."""
        phi2d = state.phi.reshape(self._nr, self._nz)
        ir = int(np.argmin(np.abs(self._r_grid_full - r_fuel_mid_cm)))
        axial = phi2d[ir, :]
        peak = axial.max() or 1.0
        return self._z_grid_full.copy(), axial / peak

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nr(self) -> int:
        return self._nr

    @property
    def nz(self) -> int:
        return self._nz

    @property
    def r_grid(self) -> np.ndarray:
        return self._r_grid_full

    @property
    def z_grid(self) -> np.ndarray:
        return self._z_grid_full


def _downsample_idx(n: int, n_d: int) -> np.ndarray:
    return np.round(np.linspace(0, n - 1, min(n_d, n))).astype(int)
