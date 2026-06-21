import math
from typing import Optional

from .config import DELAYED_NEUTRON_GROUPS, REACTOR_MODEL
from .reactivity import ThermalFeedbackInput, compute_reactivity
from .schemas import ReactorRegime, ReactorSnapshot, ThermalSnapshot


def equilibrium_precursors(neutron_population: float) -> list[float]:
    return [
        (
            group.beta
            / (REACTOR_MODEL.neutron_generation_time_seconds * group.decay_constant)
        )
        * neutron_population
        for group in DELAYED_NEUTRON_GROUPS
    ]


def classify_regime(total_pcm: float, scram_latched: bool) -> ReactorRegime:
    if scram_latched:
        return "scrammed"

    if abs(total_pcm) < 8:
        return "near-critical"

    return "supercritical" if total_pcm > 0 else "subcritical"


def calculate_period_seconds(
    previous_neutron_population: float,
    next_neutron_population: float,
    step_seconds: float,
) -> Optional[float]:
    if previous_neutron_population <= 0 or next_neutron_population <= 0:
        return None

    logarithmic_change = math.log(next_neutron_population / previous_neutron_population)

    if abs(logarithmic_change) < 1e-4:
        return None

    return step_seconds / logarithmic_change


class ReactorEngine:
    def __init__(self) -> None:
        self.rod_insertion_percent = REACTOR_MODEL.critical_rod_insertion_percent
        self.scram_latched = False
        self.time_seconds = 0.0
        self.neutron_population = 1.0
        self.precursor_concentrations = equilibrium_precursors(1.0)
        self.last_period_seconds: Optional[float] = None
        self.thermal_feedback: Optional[ThermalFeedbackInput] = None
        self.thermal_feedback_reference: Optional[ThermalFeedbackInput] = None

    def reset(self) -> None:
        self.rod_insertion_percent = REACTOR_MODEL.critical_rod_insertion_percent
        self.scram_latched = False
        self.time_seconds = 0.0
        self.neutron_population = 1.0
        self.precursor_concentrations = equilibrium_precursors(1.0)
        self.last_period_seconds = None
        self.thermal_feedback = None
        self.thermal_feedback_reference = None

    def set_thermal_feedback_snapshot(
        self,
        thermal_snapshot: ThermalSnapshot,
        *,
        reset_reference: bool = False,
    ) -> None:
        feedback = ThermalFeedbackInput(
            source=thermal_snapshot.source,
            fuel_temperature_k=thermal_snapshot.fuelTemperatureK,
            moderator_temperature_k=thermal_snapshot.moderatorTemperatureK,
            moderator_density_g_per_cm3=thermal_snapshot.moderatorDensityGPerCm3,
        )
        self.thermal_feedback = feedback
        if reset_reference:
            self.thermal_feedback_reference = feedback if feedback.active else None

    def set_rod_insertion(self, insertion_percent: float) -> None:
        if self.scram_latched:
            return

        self.rod_insertion_percent = min(max(insertion_percent, 0.0), 100.0)

    def scram(self) -> None:
        self.scram_latched = True
        self.rod_insertion_percent = 100.0

    def step(self, step_seconds: float) -> None:
        if step_seconds <= 0:
            return

        reactivity = compute_reactivity(
            self.rod_insertion_percent,
            self.scram_latched,
            self.thermal_feedback,
            self.thermal_feedback_reference,
        )
        prompt_term = (
            reactivity.totalDeltaK - REACTOR_MODEL.beta_effective
        ) / REACTOR_MODEL.neutron_generation_time_seconds

        delayed_source = 0.0
        delayed_coupling = 0.0

        for index, group in enumerate(DELAYED_NEUTRON_GROUPS):
            denominator = 1 + step_seconds * group.decay_constant
            delayed_source += (
                group.decay_constant * self.precursor_concentrations[index]
            ) / denominator
            delayed_coupling += (
                group.decay_constant
                * step_seconds
                * group.beta
                / REACTOR_MODEL.neutron_generation_time_seconds
            ) / denominator

        numerator = self.neutron_population + step_seconds * delayed_source
        denominator = 1 - step_seconds * prompt_term - step_seconds * delayed_coupling
        next_neutron_population = max(numerator / denominator, 1e-9)

        self.precursor_concentrations = [
            (
                self.precursor_concentrations[index]
                + step_seconds
                * (
                    group.beta / REACTOR_MODEL.neutron_generation_time_seconds
                )
                * next_neutron_population
            )
            / (1 + step_seconds * group.decay_constant)
            for index, group in enumerate(DELAYED_NEUTRON_GROUPS)
        ]

        self.last_period_seconds = calculate_period_seconds(
            self.neutron_population, next_neutron_population, step_seconds
        )
        self.neutron_population = next_neutron_population
        self.time_seconds += step_seconds

        if (
            not self.scram_latched
            and self.neutron_population * REACTOR_MODEL.nominal_thermal_power_mw
            >= REACTOR_MODEL.auto_scram_power_mw
        ):
            self.scram()

    def get_snapshot(self) -> ReactorSnapshot:
        reactivity = compute_reactivity(
            self.rod_insertion_percent,
            self.scram_latched,
            self.thermal_feedback,
            self.thermal_feedback_reference,
        )
        thermal_power_mw = (
            REACTOR_MODEL.nominal_thermal_power_mw * self.neutron_population
        )
        total_flux = (
            REACTOR_MODEL.nominal_flux_neutrons_per_square_centimeter_second
            * self.neutron_population
        )

        return ReactorSnapshot(
            lastPeriodSeconds=self.last_period_seconds,
            neutronPopulation=self.neutron_population,
            reactivity=reactivity,
            rodInsertionPercent=self.rod_insertion_percent,
            scramLatched=self.scram_latched,
            status=classify_regime(reactivity.totalPcm, self.scram_latched),
            thermalPowerMw=thermal_power_mw,
            timeSeconds=self.time_seconds,
            totalFlux=total_flux,
        )
