"""Singleton service for the 2D r-z transient diffusion page.

Owns a ``TransientSolver`` instance and the current simulation state.
Advances the transient on every ``get_state()`` call (if running) by as
many implicit-Euler steps as wall-clock time allows (capped at a maximum
per-request to keep HTTP latency bounded).

Thread-safety: a single ``threading.Lock`` guards all state mutations.
This is safe for single-worker FastAPI / Uvicorn (the expected localhost
deployment).
"""

from __future__ import annotations

import logging
import threading
import time

from .calibration import (
    CRITICAL_INSERTION_PERCENT,
    ESTIMATE2_H_ACTIVE_CM,
    ESTIMATE2_H_REFL_CM,
    ESTIMATE2_R_FUEL_CM,
    ESTIMATE2_R_INNER_CM,
    ESTIMATE2_R_REFL_CM,
    ROD_RADIUS_CM,
)
from .core_service import _build_calibrated_model
from .reactivity import compute_reactivity
from .schemas import (
    CoreFluxGeometry,
    CoreFluxProfile,
    TransientDiffusionState,
    TransientHistoryPoint,
)
from .transient_diffusion import (
    TRANSIENT_DT_S,
    TRANSIENT_DR_CM,
    TRANSIENT_DZ_CM,
    TransientSolver,
    TransientSolverState,
)

logger = logging.getLogger(__name__)

# How many time steps to advance per GET /state call at most.
# At dt=1 s, 4 steps = 4 s of simulation per HTTP request.
_MAX_STEPS_PER_REQUEST = 4

# Simulated seconds per wall-clock second.
_TIME_SCALE = 1.0

# History: keep last N points; record every step (dt=1 s per point).
_HISTORY_LIMIT = 200

# Radial mid-fuel reference for axial profile
_R_FUEL_MID_CM = (ESTIMATE2_R_INNER_CM + ESTIMATE2_R_FUEL_CM) / 2.0


class TransientDiffusionService:
    """Shared singleton state for the transient diffusion page."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._solver = TransientSolver(
            model=_build_calibrated_model(),
            dr=TRANSIENT_DR_CM,
            dz=TRANSIENT_DZ_CM,
            dt=TRANSIENT_DT_S,
        )
        self._running = False
        self._last_wall = time.monotonic()
        self._state: TransientSolverState | None = None
        self._k_ref: float = 1.0
        self._rod_frac: float = CRITICAL_INSERTION_PERCENT / 100.0
        self._history: list[TransientHistoryPoint] = []
        self._step_count_total: int = 0

        # Initialize synchronously at startup — first call runs solve_2d (~10 s)
        self._do_reset_locked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_reset_locked(self) -> None:
        """Reinitialize to critical state.  Caller must hold the lock."""
        rod_frac = CRITICAL_INSERTION_PERCENT / 100.0
        self._rod_frac = rod_frac
        state, k_ref = self._solver.initialize(rod_frac)
        self._k_ref = k_ref
        self._state = state
        self._running = False
        self._last_wall = time.monotonic()
        self._step_count_total = 0
        self._history = [self._history_point(state)]

    def _history_point(self, state: TransientSolverState) -> TransientHistoryPoint:
        rod_pct = self._rod_frac * 100.0
        reactivity = compute_reactivity(rod_pct, scram_latched=False)
        p_norm = self._solver.power_norm(state)
        return TransientHistoryPoint(
            timeSeconds=state.time_s,
            reactivityPcm=reactivity.totalPcm,
            powerNorm=p_norm,
        )

    def _advance_locked(self) -> None:
        """Advance simulation by wall-clock elapsed time (if running)."""
        now = time.monotonic()
        elapsed_wall = now - self._last_wall
        self._last_wall = now

        if not self._running or self._state is None:
            return

        sim_seconds = elapsed_wall * _TIME_SCALE
        n_steps = min(int(sim_seconds / TRANSIENT_DT_S), _MAX_STEPS_PER_REQUEST)
        if n_steps <= 0:
            return

        for _ in range(n_steps):
            self._state = self._solver.step(
                self._state, self._rod_frac, self._k_ref
            )
            self._step_count_total += 1
            self._history.append(self._history_point(self._state))

        self._history = self._history[-_HISTORY_LIMIT:]

    def _build_response_locked(self) -> TransientDiffusionState:
        state = self._state
        assert state is not None

        rod_pct = self._rod_frac * 100.0
        reactivity = compute_reactivity(rod_pct, scram_latched=False)
        p_norm = self._solver.power_norm(state)

        phi_d, r_d, z_d = self._solver.display_phi(state)
        nr_d, nz_d = phi_d.shape

        r_radial, phi_radial = self._solver.radial_profile(state)
        z_axial, phi_axial = self._solver.axial_profile(state, _R_FUEL_MID_CM)

        geometry = CoreFluxGeometry(
            rInnerCm=ESTIMATE2_R_INNER_CM,
            rFuelCm=ESTIMATE2_R_FUEL_CM,
            rReflCm=ESTIMATE2_R_REFL_CM,
            hActiveCm=ESTIMATE2_H_ACTIVE_CM,
            hReflCm=ESTIMATE2_H_REFL_CM,
            rodRadiusCm=ROD_RADIUS_CM,
        )

        return TransientDiffusionState(
            timeSeconds=state.time_s,
            rodInsertionPercent=rod_pct,
            running=self._running,
            reactivityPcm=reactivity.totalPcm,
            powerNorm=p_norm,
            heatmapRCm=r_d.tolist(),
            heatmapZCm=z_d.tolist(),
            heatmapPhi=[
                [round(float(v), 5) for v in phi_d[i]]
                for i in range(nr_d)
            ],
            radial=CoreFluxProfile(
                axisCm=r_radial.tolist(),
                phiNorm=[round(float(v), 5) for v in phi_radial],
            ),
            axial=CoreFluxProfile(
                axisCm=z_axial.tolist(),
                phiNorm=[round(float(v), 5) for v in phi_axial],
            ),
            history=list(self._history),
            geometry=geometry,
            dt=TRANSIENT_DT_S,
            meshDrCm=TRANSIENT_DR_CM,
            meshDzCm=TRANSIENT_DZ_CM,
            stepCount=self._step_count_total,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> TransientDiffusionState:
        with self._lock:
            self._advance_locked()
            return self._build_response_locked()

    def reset(self) -> TransientDiffusionState:
        with self._lock:
            self._do_reset_locked()
            return self._build_response_locked()

    def set_running(self, running: bool) -> TransientDiffusionState:
        with self._lock:
            self._running = running
            self._last_wall = time.monotonic()
            return self._build_response_locked()

    def set_rod_insertion(self, insertion_percent: float) -> TransientDiffusionState:
        with self._lock:
            self._rod_frac = max(0.0, min(1.0, insertion_percent / 100.0))
            self._last_wall = time.monotonic()
            return self._build_response_locked()

    def manual_step(self) -> TransientDiffusionState:
        """Advance exactly one time step regardless of running/paused state."""
        with self._lock:
            if self._state is not None:
                self._state = self._solver.step(
                    self._state, self._rod_frac, self._k_ref
                )
                self._step_count_total += 1
                self._history.append(self._history_point(self._state))
                self._history = self._history[-_HISTORY_LIMIT:]
                self._last_wall = time.monotonic()
            return self._build_response_locked()


transient_diffusion_service = TransientDiffusionService()
