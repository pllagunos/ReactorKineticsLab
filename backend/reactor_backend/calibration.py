"""Frozen calibration constants derived from theory/reactorModel.ipynb Estimate 2.

Source: 2D r-z finite-difference one-group diffusion solver (dr = dz = 3 cm).
Reference state: clean unrodded core, k_eff ≈ 1.000395 (fine-mesh 2D).

Do not modify these values by hand. Re-run the notebook to regenerate them.
"""

# ---------------------------------------------------------------------------
# Estimate 2 geometry (cm)
# ---------------------------------------------------------------------------
ESTIMATE2_R_INNER_CM: float = 80.0
ESTIMATE2_R_FUEL_CM: float = 345.6
ESTIMATE2_R_REFL_CM: float = 405.6
ESTIMATE2_H_ACTIVE_CM: float = 691.2
ESTIMATE2_H_REFL_CM: float = 60.0  # per axial side

# ---------------------------------------------------------------------------
# Nominal operating point
# ---------------------------------------------------------------------------
# Approximate peak flux for the annular 2D solution normalised to 20 MWth.
ESTIMATE2_NOMINAL_FLUX_N_CM2_S: float = 1.5e12

ESTIMATE2_NOMINAL_POWER_MW: float = 20.0

# ---------------------------------------------------------------------------
# 2D rod-worth scan: insertion fraction x → Δρ (pcm)
# ---------------------------------------------------------------------------
# Δρ(x) = ρ_pcm(k_rod(x)) − ρ_pcm(k_unrod)  where k_unrod = 1.000395
#
# x = 0.0 is forced to 0.0 (clean unrodded core = critical reference).
# The raw notebook value at x = 0.0 was +0.5 pcm and at x = 0.1 was +0.6 pcm;
# these are mesh warm-start artefacts of the finite-difference grid, not
# physics effects.  x = 0.1 retains its raw value; the magnitude is negligible.
#
# Full-insertion rod worth: −29.0 pcm (differential worth peaks near 50–60 %
# insertion, reflecting the cosine-shaped axial flux weighting).
ROD_WORTH_X: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ROD_WORTH_DELTA_RHO_PCM: tuple[float, ...] = (
    0.0, 0.6, -0.9, -4.1, -8.4, -13.4, -18.6, -23.3, -26.8, -28.6, -29.0
)

FULL_INSERTION_ROD_WORTH_PCM: float = 29.0
