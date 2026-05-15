from .calibration import CLEAN_CORE_BASE_EXCESS_PCM, ROD_WORTH_DELTA_RHO_PCM, ROD_WORTH_X
from .config import REACTOR_MODEL
from .schemas import ReactivitySnapshot


def _interpolate_rod_worth(x: float) -> float:
    """Linearly interpolate Δρ(x) from the 2D-calibrated rod-worth table.

    x is insertion fraction in [0, 1].  Returns Δρ in pcm relative to the
    clean unrodded reference (x = 0 → 0 pcm).
    """
    if x <= ROD_WORTH_X[0]:
        return ROD_WORTH_DELTA_RHO_PCM[0]
    if x >= ROD_WORTH_X[-1]:
        return ROD_WORTH_DELTA_RHO_PCM[-1]
    for i in range(len(ROD_WORTH_X) - 1):
        x0, x1 = ROD_WORTH_X[i], ROD_WORTH_X[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return ROD_WORTH_DELTA_RHO_PCM[i] + t * (ROD_WORTH_DELTA_RHO_PCM[i + 1] - ROD_WORTH_DELTA_RHO_PCM[i])
    return ROD_WORTH_DELTA_RHO_PCM[-1]


def clamp_insertion_percent(insertion_percent: float) -> float:
    return min(max(insertion_percent, 0.0), 100.0)


def compute_reactivity(
    rod_insertion_percent: float, scram_latched: bool
) -> ReactivitySnapshot:
    """Compute reactor reactivity from rod position.

    The clean unrodded core carries a small positive excess reactivity.
    Rod insertion contributes negative worth from the 2D-calibrated table.
    """
    clamped_insertion_percent = clamp_insertion_percent(rod_insertion_percent)
    x = clamped_insertion_percent / 100.0
    rod_contribution_pcm = _interpolate_rod_worth(x)
    scram_penalty_pcm = -REACTOR_MODEL.scram_shutdown_pcm if scram_latched else 0.0
    total_pcm = CLEAN_CORE_BASE_EXCESS_PCM + rod_contribution_pcm + scram_penalty_pcm

    return ReactivitySnapshot(
        baseExcessPcm=CLEAN_CORE_BASE_EXCESS_PCM,
        dollars=total_pcm / REACTOR_MODEL.beta_effective_pcm,
        rodContributionPcm=rod_contribution_pcm,
        rodInsertionPercent=clamped_insertion_percent,
        scramPenaltyPcm=scram_penalty_pcm,
        totalDeltaK=total_pcm * 1e-5,
        totalPcm=total_pcm,
    )
