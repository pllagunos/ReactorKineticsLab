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
# Combined control/shutdown bank design
# ---------------------------------------------------------------------------
# Central equivalent shutdown-bank geometry selected from 2D sweep:
# - rod radius widened from 10 cm to 50 cm (equivalent absorber volume)
# - stronger effective absorber increment in the inserted zone
ROD_RADIUS_CM: float = 50.0
ROD_DELTA_SIGMA_A_MAX_CM_INV: float = 0.25

# Clean-core operating point from fine-mesh 2D solve (unrodded, x=0).
CLEAN_CORE_KEFF: float = 1.000395
CLEAN_CORE_BASE_EXCESS_PCM: float = 39.4

# ---------------------------------------------------------------------------
# 2D rod-worth scan: insertion fraction x → Δρ (pcm)
# ---------------------------------------------------------------------------
# Δρ(x) = ρ_pcm(k_rod(x)) − ρ_pcm(k_unrod)  where k_unrod = 1.000395
#
# x = 0.0 is forced to 0.0 by definition (reference against itself).
# This table is for the combined control/shutdown bank that reaches a modest
# shutdown margin at full insertion while still giving an intermediate
# operating critical point (~32 % insertion).
#
# Full-insertion rod worth: −159.6 pcm relative to unrodded.
ROD_WORTH_X: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ROD_WORTH_DELTA_RHO_PCM: tuple[float, ...] = (
    0.0, -4.7, -16.3, -34.7, -58.1, -84.7, -111.9, -136.2, -152.5, -158.9, -159.6
)

FULL_INSERTION_ROD_WORTH_PCM: float = 159.6
CRITICAL_INSERTION_PERCENT: float = 32.0
FULL_INSERTION_TOTAL_PCM: float = -120.2
