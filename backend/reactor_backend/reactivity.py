from .config import REACTOR_MODEL
from .rod_worth import ROD_WORTH_TABLE
from .schemas import ReactivitySnapshot


def _interpolate_rod_worth(x: float) -> float:
    """Linearly interpolate Δρ(x) from the OpenMC rod-worth table.

    x is insertion fraction in [0, 1].  Returns Δρ in pcm relative to the
    clean unrodded reference (x = 0 → 0 pcm).
    """
    return ROD_WORTH_TABLE.interpolate_rod_worth_pcm(x)


def clamp_insertion_percent(insertion_percent: float) -> float:
    return min(max(insertion_percent, 0.0), 100.0)


def compute_reactivity(
    rod_insertion_percent: float, scram_latched: bool
) -> ReactivitySnapshot:
    """Compute reactor reactivity from rod position.

    The clean unrodded core carries the OpenMC CE rod-scan excess
    reactivity. Rod insertion contributes negative worth from the same table.
    """
    clamped_insertion_percent = clamp_insertion_percent(rod_insertion_percent)
    x = clamped_insertion_percent / 100.0
    base_excess_pcm = ROD_WORTH_TABLE.clean_excess_pcm
    rod_contribution_pcm = _interpolate_rod_worth(x)
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
