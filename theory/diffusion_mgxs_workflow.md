# OpenMC MGXS to Multigroup Diffusion Workflow

## Scope

The concentric diffusion model uses resolved OpenMC cell-wise multigroup
constants. It does not homogenize the fuel rings into one core material and it
does not solve higher-order transport equations.

The production workflow has four distinct stages:

1. OpenMC continuous-energy calculation and MGXS tallying.
2. OpenMC multigroup transport validation.
3. Conversion of the canonical MGXS export into diffusion regions and a
   boundary-fitted finite-volume mesh.
4. Group-wise diffusion solution and optional persistence of the prepared
   matrices and clean-core result.

SPH factors, discontinuity factors, and rodded MGXS are intentionally deferred
until the clean unrodded diffusion calculation is stable and verified.

## Scattering Contract

Two scattering representations are required from the same continuous-energy
run:

- The canonical diffusion export uses non-`nu`, uncorrected scattering
  (`scatter_correction = null`). The adapter prefers
  `consistent scatter matrix` and falls back to `scatter matrix`.
- OpenMC multigroup validation uses a supplemental P0-corrected consistent
  scattering library. This correction is validation-only and is not written
  into the canonical diffusion JSON.

OpenMC's P0 scatter correction modifies the within-group scattering treatment
used by its multigroup transport calculation. It is separate from the
transport cross section used to calculate the diffusion coefficient. The
diffusion model therefore retains

```text
D_g = 1 / (3 Sigma_tr,g)
```

while rejecting a canonical export whose scattering matrix declares a P0
correction.

Only the zeroth Legendre moment is used by diffusion. If an export contains
higher moments, the adapter records the source order and discards every moment
above P0.

## Resolved Geometry

The selected export directory is the source of truth:

```text
group_sweep/group_<N>/
  reactor_run/model.xml
  outputs/mgxs_constants.json
```

Every exported OpenMC cell remains a separate diffusion region. The adapter
requires the resolved fuel-ring cells, central moderator channel, core coolant
and moderator, outer moderator, reflector, and parked control rod when present.
Missing or mismatched XML/MGXS domains are errors; there is no supercell
fallback.

The radial mesh is fitted exactly to all XML-derived material boundaries.
Within each interval, the default target spacings are:

| Region | Radial target spacing |
| --- | ---: |
| Fuel rings | 0.1 cm |
| Central channel and core coolant | 1 cm |
| Outer moderator | 5 cm |
| Reflector and extrapolation region | 10 cm |

The default axial target spacing is 10 cm. All values are user-overridable.
Subdividing an interval never moves a material boundary.

## Group-Wise Diffusion Solve

For each energy group, the solver assembles one sparse spatial loss operator:

```text
A_g phi_g =
    chi_g / k * sum_f(nu Sigma_f,f phi_f)
  + sum_(f != g)(Sigma_s,f->g phi_f)
```

`A_g` contains leakage, absorption, and outscatter from group `g`. The
within-group scattering term is retained on the loss side through the removal
definition rather than added as a source.

The sparse spatial operator for each group is factorized independently. The
fission and scattering sources are evaluated directly from cell-wise arrays;
the production solver does not assemble a monolithic
`(groups * cells) x (groups * cells)` matrix. Inner multigroup
Gauss-Seidel/source iterations provide the initial scattering update. A
matrix-free GMRES correction, preconditioned by the same per-group factors,
accelerates highly scattering cases without assembling a global matrix. Outer
power iterations converge the eigenvalue and fission-source shape. The saved
result includes a volume-weighted final balance residual.

This split is required for practical higher-group calculations. A monolithic
sparse LU factorization can produce substantial fill-in, so its time and memory
cost do not scale linearly with the number of unknowns. The global formulation
is retained only as a small-problem reference for verification.

All eigenvalue normalization and convergence integrals use cylindrical cell
volumes.

## Persistent Cache

`backend/openmc_to_diffusion.py --prepare-cache` prepares a reusable cache under
`backend/.cache/openmc_diffusion` by default. The fingerprint includes:

- canonical MGXS JSON contents;
- colocated `model.xml` contents;
- mesh-spacing configuration;
- solver tolerances and algorithm schema.

The cache persists:

- radial and axial mesh edges;
- cell-wise diffusion, absorption, fission, spectrum, and P0 scattering arrays;
- one CSR spatial matrix per energy group;
- the converged clean-core eigenvalue and flux.

SuperLU factorization objects are process-local and are not serialized. A
backend process can load prepared CSR matrices and the clean result, then build
in-memory factorizations only when a new solve needs them. A missing or stale
cache is rebuilt synchronously. The cache is an optimization, not an alternate
source of nuclear data.

## Verification Milestone

Before introducing SPH corrections, acceptance is based on:

- finite, positive, deterministic clean-core eigenvalues;
- convergence without negative or non-finite states;
- agreement between group-wise and global reference formulations on a small
  mesh;
- coherent mesh and group-count trends;
- practical 16-group runtime on the boundary-fitted mesh.

OpenMC continuous-energy and multigroup `k_eff` values are contextual
references at this stage. Tight pcm agreement is not required until the
uncorrected diffusion workflow is stable enough to support a controlled SPH
calibration.

# Results
| Groups | MG OpenMC error vs CE | Diffusion error vs CE | Approx. diffusion bias vs MG |
| -----: | --------------------: | --------------------: | ---------------------------: |
|      1 |             +1908 pcm |             +1800 pcm |                     −108 pcm |
|      2 |              +806 pcm |             +2000 pcm |                    +1194 pcm |
|      4 |              +426 pcm |               −94 pcm |                     −520 pcm |
|      8 |              +166 pcm |              −366 pcm |                     −532 pcm |
|     16 |               +66 pcm |              −400 pcm |                     −466 pcm |
