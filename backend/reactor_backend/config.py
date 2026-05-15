from dataclasses import dataclass

from .calibration import (
    CRITICAL_INSERTION_PERCENT,
    ESTIMATE2_H_ACTIVE_CM,
    ESTIMATE2_NOMINAL_FLUX_N_CM2_S,
    ESTIMATE2_NOMINAL_POWER_MW,
    ESTIMATE2_R_FUEL_CM,
    ESTIMATE2_R_INNER_CM,
    FULL_INSERTION_ROD_WORTH_PCM,
)


@dataclass(frozen=True)
class DelayedNeutronGroup:
    beta: float
    decay_constant: float


@dataclass(frozen=True)
class CoreGeometryConfig:
    active_height_meters: float
    inner_radius_meters: float
    outer_radius_meters: float


@dataclass(frozen=True)
class ReactorModelConfig:
    auto_scram_power_mw: float
    beta_effective: float
    beta_effective_pcm: float
    core_geometry: CoreGeometryConfig
    critical_rod_insertion_percent: float
    nominal_flux_neutrons_per_square_centimeter_second: float
    nominal_thermal_power_mw: float
    neutron_generation_time_seconds: float
    scram_shutdown_pcm: float
    total_control_rod_worth_pcm: float


@dataclass(frozen=True)
class SimulationTuningConfig:
    history_point_limit: int
    history_sample_seconds: float
    integrator_step_seconds: float
    max_wall_step_seconds: float
    poll_interval_ms: int
    time_scale: float


DELAYED_NEUTRON_GROUPS = (
    DelayedNeutronGroup(beta=0.00025, decay_constant=0.0124),
    DelayedNeutronGroup(beta=0.00138, decay_constant=0.0305),
    DelayedNeutronGroup(beta=0.00122, decay_constant=0.111),
    DelayedNeutronGroup(beta=0.00264, decay_constant=0.301),
    DelayedNeutronGroup(beta=0.00075, decay_constant=1.14),
    DelayedNeutronGroup(beta=0.00027, decay_constant=3.01),
)

REACTOR_MODEL = ReactorModelConfig(
    # Scram at 1.3 × nominal power (Estimate 2 baseline: 20 MWth)
    auto_scram_power_mw=26,
    beta_effective=0.00651,
    beta_effective_pcm=651,
    # Estimate 2 geometry (2D-critical clean core, from theory/reactorModel.ipynb)
    core_geometry=CoreGeometryConfig(
        active_height_meters=ESTIMATE2_H_ACTIVE_CM / 100.0,
        inner_radius_meters=ESTIMATE2_R_INNER_CM / 100.0,
        outer_radius_meters=ESTIMATE2_R_FUEL_CM / 100.0,
    ),
    # Combined control/shutdown bank is tuned so normal operation is near
    # critical around 32% insertion and fully inserted gives negative margin.
    critical_rod_insertion_percent=CRITICAL_INSERTION_PERCENT,
    nominal_flux_neutrons_per_square_centimeter_second=ESTIMATE2_NOMINAL_FLUX_N_CM2_S,
    nominal_thermal_power_mw=ESTIMATE2_NOMINAL_POWER_MW,
    neutron_generation_time_seconds=5e-4,
    scram_shutdown_pcm=450,
    total_control_rod_worth_pcm=FULL_INSERTION_ROD_WORTH_PCM,
)

SIMULATION_TUNING = SimulationTuningConfig(
    history_point_limit=240,
    history_sample_seconds=0.25,
    integrator_step_seconds=0.02,
    max_wall_step_seconds=0.2,
    poll_interval_ms=100,
    time_scale=8,
)
