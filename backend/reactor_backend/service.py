import threading
import time

from .config import REACTOR_MODEL, SIMULATION_TUNING
from .engine import ReactorEngine
from .schemas import CoreGeometry, HistoryPoint, ReactorModel, SimulationState, SimulationTuning


def to_history_point(time_seconds: float, reactivity_pcm: float, total_flux: float, thermal_power_mw: float) -> HistoryPoint:
    return HistoryPoint(
        timeSeconds=time_seconds,
        reactivityPcm=reactivity_pcm,
        totalFlux=total_flux,
        thermalPowerMw=thermal_power_mw,
    )


class SimulationService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = ReactorModel(
            autoScramPowerMw=REACTOR_MODEL.auto_scram_power_mw,
            betaEffective=REACTOR_MODEL.beta_effective,
            betaEffectivePcm=REACTOR_MODEL.beta_effective_pcm,
            coreGeometry=CoreGeometry(
                activeHeightMeters=REACTOR_MODEL.core_geometry.active_height_meters,
                innerRadiusMeters=REACTOR_MODEL.core_geometry.inner_radius_meters,
                outerRadiusMeters=REACTOR_MODEL.core_geometry.outer_radius_meters,
            ),
            criticalRodInsertionPercent=REACTOR_MODEL.critical_rod_insertion_percent,
            nominalFluxNeutronsPerSquareCentimeterSecond=REACTOR_MODEL.nominal_flux_neutrons_per_square_centimeter_second,
            nominalThermalPowerMw=REACTOR_MODEL.nominal_thermal_power_mw,
            neutronGenerationTimeSeconds=REACTOR_MODEL.neutron_generation_time_seconds,
            scramShutdownPcm=REACTOR_MODEL.scram_shutdown_pcm,
            totalControlRodWorthPcm=REACTOR_MODEL.total_control_rod_worth_pcm,
        )
        self._tuning = SimulationTuning(
            historyPointLimit=SIMULATION_TUNING.history_point_limit,
            historySampleSeconds=SIMULATION_TUNING.history_sample_seconds,
            integratorStepSeconds=SIMULATION_TUNING.integrator_step_seconds,
            maxWallStepSeconds=SIMULATION_TUNING.max_wall_step_seconds,
            pollIntervalMs=SIMULATION_TUNING.poll_interval_ms,
            timeScale=SIMULATION_TUNING.time_scale,
        )
        self._engine = ReactorEngine()
        self._running = True
        self._history_accumulator = 0.0
        self._last_wall_time = time.monotonic()
        self._history = [self._current_history_point()]

    def _current_history_point(self) -> HistoryPoint:
        snapshot = self._engine.get_snapshot()
        return to_history_point(
            snapshot.timeSeconds,
            snapshot.reactivity.totalPcm,
            snapshot.totalFlux,
            snapshot.thermalPowerMw,
        )

    def _push_history_point(self) -> None:
        self._history.append(self._current_history_point())
        self._history = self._history[-SIMULATION_TUNING.history_point_limit :]

    def _advance_locked(self) -> None:
        now = time.monotonic()
        elapsed_wall_seconds = now - self._last_wall_time
        self._last_wall_time = now

        if not self._running:
            return

        elapsed_wall_seconds = min(
            elapsed_wall_seconds, SIMULATION_TUNING.max_wall_step_seconds
        )
        simulated_seconds_remaining = elapsed_wall_seconds * SIMULATION_TUNING.time_scale

        while simulated_seconds_remaining > 0:
            step_seconds = min(
                simulated_seconds_remaining, SIMULATION_TUNING.integrator_step_seconds
            )
            self._engine.step(step_seconds)
            simulated_seconds_remaining -= step_seconds
            self._history_accumulator += step_seconds

            if self._history_accumulator >= SIMULATION_TUNING.history_sample_seconds:
                self._history_accumulator -= SIMULATION_TUNING.history_sample_seconds
                self._push_history_point()

    def _build_state_locked(self) -> SimulationState:
        return SimulationState(
            history=list(self._history),
            model=self._model,
            running=self._running,
            snapshot=self._engine.get_snapshot(),
            tuning=self._tuning,
        )

    def get_state(self) -> SimulationState:
        with self._lock:
            self._advance_locked()
            return self._build_state_locked()

    def reset(self) -> SimulationState:
        with self._lock:
            self._engine.reset()
            self._running = True
            self._history_accumulator = 0.0
            self._last_wall_time = time.monotonic()
            self._history = [self._current_history_point()]
            return self._build_state_locked()

    def scram(self) -> SimulationState:
        with self._lock:
            self._advance_locked()
            self._engine.scram()
            self._push_history_point()
            return self._build_state_locked()

    def set_rod_insertion(self, insertion_percent: float) -> SimulationState:
        with self._lock:
            self._advance_locked()
            self._engine.set_rod_insertion(insertion_percent)
            return self._build_state_locked()

    def set_running(self, running: bool) -> SimulationState:
        with self._lock:
            self._advance_locked()
            self._running = running
            self._last_wall_time = time.monotonic()
            return self._build_state_locked()


simulation_service = SimulationService()
