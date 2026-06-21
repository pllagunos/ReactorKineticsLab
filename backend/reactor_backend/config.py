from dataclasses import dataclass

from .calibration import (
    ESTIMATE2_H_ACTIVE_CM,
    ESTIMATE2_NOMINAL_FLUX_N_CM2_S,
    ESTIMATE2_NOMINAL_POWER_MW,
    ESTIMATE2_R_FUEL_CM,
    ESTIMATE2_R_INNER_CM,
)
from .kinetics_reference import KINETICS_REFERENCE
from .rod_worth import ROD_WORTH_TABLE


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
    thermal_update_seconds: float
    max_wall_step_seconds: float
    poll_interval_ms: int
    time_scale: float


DELAYED_NEUTRON_GROUPS = tuple(
    DelayedNeutronGroup(
        beta=group.beta,
        decay_constant=group.decay_constant,
    )
    for group in KINETICS_REFERENCE.delayed_groups
)

REACTOR_MODEL = ReactorModelConfig(
    # Scram at 1.3 × nominal power (Estimate 2 baseline: 20 MWth)
    auto_scram_power_mw=1000,
    beta_effective=KINETICS_REFERENCE.beta_effective,
    beta_effective_pcm=KINETICS_REFERENCE.beta_effective_pcm,
    # Estimate 2 geometry (2D-critical clean core, from theory/reactorModel.ipynb)
    core_geometry=CoreGeometryConfig(
        active_height_meters=ESTIMATE2_H_ACTIVE_CM / 100.0,
        inner_radius_meters=ESTIMATE2_R_INNER_CM / 100.0,
        outer_radius_meters=ESTIMATE2_R_FUEL_CM / 100.0,
    ),
    # Rod-worth metadata is loaded from the OpenMC CE rod scan reference.
    critical_rod_insertion_percent=ROD_WORTH_TABLE.critical_insertion_percent,
    nominal_flux_neutrons_per_square_centimeter_second=ESTIMATE2_NOMINAL_FLUX_N_CM2_S,
    nominal_thermal_power_mw=ESTIMATE2_NOMINAL_POWER_MW,
    neutron_generation_time_seconds=KINETICS_REFERENCE.neutron_generation_time_seconds,
    scram_shutdown_pcm=450,
    total_control_rod_worth_pcm=ROD_WORTH_TABLE.full_insertion_worth_pcm,
)

SIMULATION_TUNING = SimulationTuningConfig(
    history_point_limit=1200,
    history_sample_seconds=0.25,
    integrator_step_seconds=0.02,
    thermal_update_seconds=0.1,
    max_wall_step_seconds=0.2,
    poll_interval_ms=100,
    time_scale=8,
)
