"""Diffusion-backed core flux service.

Wraps ``diffusion.solve_2d`` behind an in-process cache so repeated requests
for the same rod insertion state do not re-run the eigenvalue solve.
The cached key is (rod_fraction_rounded_2dp, dr_cm, dz_cm).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .calibration import (
    D_FUEL_CM,
    D_MOD_INNER_CM,
    D_REFL_CM,
    ESTIMATE2_H_ACTIVE_CM,
    ESTIMATE2_H_REFL_CM,
    ESTIMATE2_R_FUEL_CM,
    ESTIMATE2_R_INNER_CM,
    ESTIMATE2_R_REFL_CM,
    NU_SIGMA_F_FUEL_CM_INV,
    ROD_DELTA_SIGMA_A_MAX_CM_INV,
    ROD_RADIUS_CM,
    SIGMA_A_FUEL_CM_INV,
    SIGMA_A_MOD_INNER_CM_INV,
    SIGMA_A_REFL_CM_INV,
)
from .diffusion import AnnularModel, Model2D, Region, solve_2d
from .schemas import (
    CoreFluxGeometry,
    CoreFluxMetadata,
    CoreFluxProfile,
    CoreFluxResponse,
)

logger = logging.getLogger(__name__)

# Display resolution for the heatmap (downsampled from solver mesh)
_NR_DISPLAY = 60
_NZ_DISPLAY = 90

_MESH_DR_CM = 3.0
_MESH_DZ_CM = 3.0


def _build_calibrated_model() -> Model2D:
    inner_mod = Region(
        name="inner_D2O_moderator",
        D=D_MOD_INNER_CM,
        Sigma_a=SIGMA_A_MOD_INNER_CM_INV,
        nuSigma_f=0.0,
    )
    fuel = Region(
        name="homogenized_fuel_annulus",
        D=D_FUEL_CM,
        Sigma_a=SIGMA_A_FUEL_CM_INV,
        nuSigma_f=NU_SIGMA_F_FUEL_CM_INV,
    )
    reflector = Region(
        name="outer_D2O_reflector",
        D=D_REFL_CM,
        Sigma_a=SIGMA_A_REFL_CM_INV,
        nuSigma_f=0.0,
    )
    base = AnnularModel(
        R_inner=ESTIMATE2_R_INNER_CM,
        R_fuel=ESTIMATE2_R_FUEL_CM,
        R_refl=ESTIMATE2_R_REFL_CM,
        mod_inner=inner_mod,
        fuel=fuel,
        reflector=reflector,
    )
    return Model2D(
        base=base,
        H=ESTIMATE2_H_ACTIVE_CM,
        H_refl=ESTIMATE2_H_REFL_CM,
        r_rod=ROD_RADIUS_CM,
        dSa_rod=ROD_DELTA_SIGMA_A_MAX_CM_INV,
    )


_MODEL: Model2D = _build_calibrated_model()

# Simple dict cache: (rod_frac_2dp, dr, dz) → raw solve_2d result dict
_cache: dict[tuple[float, float, float], dict[str, Any]] = {}


def _get_solve(rod_frac: float, dr: float, dz: float) -> tuple[dict[str, Any], bool]:
    key = (round(rod_frac, 2), dr, dz)
    if key in _cache:
        return _cache[key], True

    t0 = time.perf_counter()
    result = solve_2d(_MODEL, dr=dr, dz=dz, x_insert=rod_frac)
    elapsed = time.perf_counter() - t0
    logger.info(
        "solve_2d completed in %.2f s  (x=%.2f, dr=%.1f, dz=%.1f, k_eff=%.6f, iter=%d)",
        elapsed,
        rod_frac,
        dr,
        dz,
        result["k_eff"],
        result["iterations"],
    )
    _cache[key] = result
    return result, False


def _downsample_indices(n: int, n_display: int) -> np.ndarray:
    return np.round(np.linspace(0, n - 1, min(n_display, n))).astype(int)


def get_core_flux(rod_insertion_percent: float) -> CoreFluxResponse:
    """Return a display-ready diffusion flux payload for the Core page.

    Parameters
    ----------
    rod_insertion_percent:
        Current rod insertion in [0, 100].
    """
    rod_frac = max(0.0, min(1.0, rod_insertion_percent / 100.0))
    sol, from_cache = _get_solve(rod_frac, _MESH_DR_CM, _MESH_DZ_CM)

    phi: np.ndarray = sol["phi"]          # (Nr, Nz), normalised to peak=1
    r_grid: np.ndarray = sol["r_grid"]    # (Nr,) cm
    z_grid: np.ndarray = sol["z_grid"]    # (Nz,) cm
    nr: int = sol["Nr"]
    nz: int = sol["Nz"]

    # Downsample for heatmap display
    r_idx = _downsample_indices(nr, _NR_DISPLAY)
    z_idx = _downsample_indices(nz, _NZ_DISPLAY)
    phi_display = phi[np.ix_(r_idx, z_idx)]
    r_display = r_grid[r_idx]
    z_display = z_grid[z_idx]

    nr_d, nz_d = phi_display.shape

    # Midplane radial profile (z ≈ 0)
    iz_mid = int(np.argmin(np.abs(z_grid)))
    radial_phi = phi[:, iz_mid]
    radial_max = radial_phi.max() or 1.0

    # Axial profile at radial midpoint of the fuel annulus
    r_fuel_mid = (ESTIMATE2_R_INNER_CM + ESTIMATE2_R_FUEL_CM) / 2.0
    ir_fuel = int(np.argmin(np.abs(r_grid - r_fuel_mid)))
    axial_phi = phi[ir_fuel, :]
    axial_max = axial_phi.max() or 1.0

    geometry = CoreFluxGeometry(
        rInnerCm=ESTIMATE2_R_INNER_CM,
        rFuelCm=ESTIMATE2_R_FUEL_CM,
        rReflCm=ESTIMATE2_R_REFL_CM,
        hActiveCm=ESTIMATE2_H_ACTIVE_CM,
        hReflCm=ESTIMATE2_H_REFL_CM,
        rodRadiusCm=ROD_RADIUS_CM,
    )

    metadata = CoreFluxMetadata(
        rodInsertionPercent=rod_insertion_percent,
        kEff=float(sol["k_eff"]),
        iterations=int(sol["iterations"]),
        cached=from_cache,
        meshDrCm=_MESH_DR_CM,
        meshDzCm=_MESH_DZ_CM,
        displayNr=nr_d,
        displayNz=nz_d,
    )

    return CoreFluxResponse(
        heatmapRCm=r_display.tolist(),
        heatmapZCm=z_display.tolist(),
        heatmapPhi=[[round(float(v), 5) for v in phi_display[i]] for i in range(nr_d)],
        radial=CoreFluxProfile(
            axisCm=r_grid.tolist(),
            phiNorm=[round(float(v) / radial_max, 5) for v in radial_phi],
        ),
        axial=CoreFluxProfile(
            axisCm=z_grid.tolist(),
            phiNorm=[round(float(v) / axial_max, 5) for v in axial_phi],
        ),
        geometry=geometry,
        metadata=metadata,
    )


def compute_axial_power_fractions_8(rod_insertion_percent: float) -> list[float]:
    """Compute 8 equal-height axial node power fractions from the cached diffusion solve.

    Returns 8 fractions ordered **top-to-bottom** (node 1 = inlet at top, as expected
    by the Modelica DynamicPipe for downflow) that sum to 1.0.
    """
    rod_frac = max(0.0, min(1.0, rod_insertion_percent / 100.0))
    sol, _ = _get_solve(rod_frac, _MESH_DR_CM, _MESH_DZ_CM)

    phi: np.ndarray = sol["phi"]        # (Nr, Nz), normalised to peak = 1
    r_grid: np.ndarray = sol["r_grid"]  # (Nr,) cm

    # Same radial position as get_core_flux: mid-radius of the fuel annulus
    r_fuel_mid = (ESTIMATE2_R_INNER_CM + ESTIMATE2_R_FUEL_CM) / 2.0
    ir_fuel = int(np.argmin(np.abs(r_grid - r_fuel_mid)))
    axial_phi = phi[ir_fuel, :]  # (Nz,), indexed bottom (z=0) → top (z=H)

    nz = len(axial_phi)
    fracs_bottom_to_top: list[float] = []
    for i in range(8):
        lo = i * nz // 8
        hi = (i + 1) * nz // 8
        fracs_bottom_to_top.append(float(np.mean(axial_phi[lo:hi])))

    # Reverse so index 0 = top = inlet for downflow core
    fracs = fracs_bottom_to_top[::-1]

    total = sum(fracs)
    if total > 0.0:
        fracs = [f / total for f in fracs]
    else:
        fracs = [0.125] * 8

    return fracs
