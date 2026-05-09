import math

from .config import REACTOR_MODEL
from .schemas import ReactivitySnapshot


def cumulative_worth_shape(insertion_fraction: float) -> float:
    return insertion_fraction - math.sin(2 * math.pi * insertion_fraction) / (2 * math.pi)


def clamp_insertion_percent(insertion_percent: float) -> float:
    return min(max(insertion_percent, 0.0), 100.0)


def compute_reactivity(
    rod_insertion_percent: float, scram_latched: bool
) -> ReactivitySnapshot:
    clamped_insertion_percent = clamp_insertion_percent(rod_insertion_percent)
    insertion_fraction = clamped_insertion_percent / 100.0
    base_excess_pcm = REACTOR_MODEL.total_control_rod_worth_pcm * cumulative_worth_shape(
        REACTOR_MODEL.critical_rod_insertion_percent / 100.0
    )
    rod_contribution_pcm = (
        -REACTOR_MODEL.total_control_rod_worth_pcm
        * cumulative_worth_shape(insertion_fraction)
    )
    scram_penalty_pcm = -REACTOR_MODEL.scram_shutdown_pcm if scram_latched else 0.0
    total_pcm = base_excess_pcm + rod_contribution_pcm + scram_penalty_pcm

    return ReactivitySnapshot(
        baseExcessPcm=base_excess_pcm,
        dollars=total_pcm / REACTOR_MODEL.beta_effective_pcm,
        rodContributionPcm=rod_contribution_pcm,
        rodInsertionPercent=clamped_insertion_percent,
        scramPenaltyPcm=scram_penalty_pcm,
        totalDeltaK=total_pcm * 1e-5,
        totalPcm=total_pcm,
    )
