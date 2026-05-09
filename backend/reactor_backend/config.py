from dataclasses import dataclass


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
    auto_scram_power_mw=325,
    beta_effective=0.00651,
    beta_effective_pcm=651,
    core_geometry=CoreGeometryConfig(
        active_height_meters=5.8,
        inner_radius_meters=0.8,
        outer_radius_meters=2.2,
    ),
    critical_rod_insertion_percent=50,
    nominal_flux_neutrons_per_square_centimeter_second=2.4e13,
    nominal_thermal_power_mw=250,
    neutron_generation_time_seconds=5e-4,
    scram_shutdown_pcm=450,
    total_control_rod_worth_pcm=700,
)

SIMULATION_TUNING = SimulationTuningConfig(
    history_point_limit=240,
    history_sample_seconds=0.25,
    integrator_step_seconds=0.02,
    max_wall_step_seconds=0.2,
    poll_interval_ms=100,
    time_scale=8,
)
