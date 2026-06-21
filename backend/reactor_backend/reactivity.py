from dataclasses import dataclass
from typing import Optional

from .config import REACTOR_MODEL
from .reactivity_coefficients import REACTIVITY_COEFFICIENTS
from .rod_worth import ROD_WORTH_TABLE
from .schemas import ReactivitySnapshot


@dataclass(frozen=True)
class ThermalFeedbackInput:
    source: str
    fuel_temperature_k: Optional[float]
    moderator_temperature_k: Optional[float]
    moderator_density_g_per_cm3: Optional[float]

    @property
    def active(self) -> bool:
        return (
            self.source == "fmu"
            and self.fuel_temperature_k is not None
            and self.moderator_temperature_k is not None
            and self.moderator_density_g_per_cm3 is not None
        )


@dataclass(frozen=True)
class ThermalFeedbackTerms:
    fuel_temperature_pcm: float = 0.0
    moderator_temperature_pcm: float = 0.0
    moderator_density_pcm: float = 0.0
    applied: bool = False

    @property
    def total_pcm(self) -> float:
        return (
            self.fuel_temperature_pcm
            + self.moderator_temperature_pcm
            + self.moderator_density_pcm
        )


def _interpolate_rod_worth(x: float) -> float:
    """Linearly interpolate Δρ(x) from the OpenMC rod-worth table.

    x is insertion fraction in [0, 1].  Returns Δρ in pcm relative to the
    clean unrodded reference (x = 0 → 0 pcm).
    """
    return ROD_WORTH_TABLE.interpolate_rod_worth_pcm(x)


def clamp_insertion_percent(insertion_percent: float) -> float:
    return min(max(insertion_percent, 0.0), 100.0)


def compute_thermal_feedback(
    thermal_feedback: Optional[ThermalFeedbackInput],
    reference_feedback: Optional[ThermalFeedbackInput] = None,
) -> ThermalFeedbackTerms:
    if thermal_feedback is None or not thermal_feedback.active:
        return ThermalFeedbackTerms()

    coefficients = REACTIVITY_COEFFICIENTS
    reference = (
        reference_feedback
        if reference_feedback is not None and reference_feedback.active
        else ThermalFeedbackInput(
            source="fmu",
            fuel_temperature_k=coefficients.fuel_temperature_base_k,
            moderator_temperature_k=coefficients.moderator_temperature_base_k,
            moderator_density_g_per_cm3=coefficients.moderator_density_base_g_per_cm3,
        )
    )
    assert reference.fuel_temperature_k is not None
    assert reference.moderator_temperature_k is not None
    assert reference.moderator_density_g_per_cm3 is not None
    assert thermal_feedback.fuel_temperature_k is not None
    assert thermal_feedback.moderator_temperature_k is not None
    assert thermal_feedback.moderator_density_g_per_cm3 is not None

    return ThermalFeedbackTerms(
        fuel_temperature_pcm=coefficients.fuel_temperature_pcm_per_k
        * (thermal_feedback.fuel_temperature_k - reference.fuel_temperature_k),
        moderator_temperature_pcm=coefficients.moderator_temperature_pcm_per_k
        * (thermal_feedback.moderator_temperature_k - reference.moderator_temperature_k),
        moderator_density_pcm=coefficients.moderator_density_pcm_per_g_per_cm3
        * (
            thermal_feedback.moderator_density_g_per_cm3
            - reference.moderator_density_g_per_cm3
        ),
        applied=True,
    )


def compute_reactivity(
    rod_insertion_percent: float,
    scram_latched: bool,
    thermal_feedback: Optional[ThermalFeedbackInput] = None,
    reference_feedback: Optional[ThermalFeedbackInput] = None,
) -> ReactivitySnapshot:
    """Compute reactor reactivity from rod position.

    The clean unrodded core carries the OpenMC CE rod-scan excess
    reactivity. Rod insertion contributes negative worth from the same table.
    FMI-only thermal feedback is added when effective fuel temperature,
    moderator temperature, and moderator density are available.
    """
    clamped_insertion_percent = clamp_insertion_percent(rod_insertion_percent)
    x = clamped_insertion_percent / 100.0
    base_excess_pcm = ROD_WORTH_TABLE.clean_excess_pcm
    rod_contribution_pcm = _interpolate_rod_worth(x)
    scram_penalty_pcm = -REACTOR_MODEL.scram_shutdown_pcm if scram_latched else 0.0
    feedback = compute_thermal_feedback(thermal_feedback, reference_feedback)
    total_pcm = (
        base_excess_pcm
        + rod_contribution_pcm
        + scram_penalty_pcm
        + feedback.total_pcm
    )

    return ReactivitySnapshot(
        baseExcessPcm=base_excess_pcm,
        dollars=total_pcm / REACTOR_MODEL.beta_effective_pcm,
        fuelTemperatureFeedbackPcm=feedback.fuel_temperature_pcm,
        moderatorDensityFeedbackPcm=feedback.moderator_density_pcm,
        moderatorTemperatureFeedbackPcm=feedback.moderator_temperature_pcm,
        rodContributionPcm=rod_contribution_pcm,
        rodInsertionPercent=clamped_insertion_percent,
        scramPenaltyPcm=scram_penalty_pcm,
        thermalFeedbackApplied=feedback.applied,
        thermalFeedbackPcm=feedback.total_pcm,
        totalDeltaK=total_pcm * 1e-5,
        totalPcm=total_pcm,
    )
