# OpenMC MGXS to Multigroup Diffusion Workflow

## Scope

The concentric diffusion model uses resolved OpenMC cell-wise multigroup
constants. It does not homogenize the fuel rings into one core material and it
does not solve higher-order transport equations.

The production workflow has five distinct stages:

1. OpenMC continuous-energy calculation and MGXS tallying.
2. OpenMC multigroup transport validation.
3. Conversion of the canonical MGXS export into diffusion regions and a
   boundary-fitted finite-volume mesh.
4. CE-referenced SPH fitting on a fixed, qualified diffusion mesh.
5. Group-wise diffusion solution and persistence of the factors, prepared
   matrices, and clean-core result.

The first online model is four-group and clean-core only. Discontinuity
factors, rodded MGXS, and rod-dependent SPH remain out of scope.

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

## Mesh Qualification

SPH factors are mesh dependent, so the production mesh is selected before
fitting. The tested meshes were:

| Mesh | Fuel/coolant/moderator/reflector radial targets | Axial target | Cells | Runtime |
| --- | --- | ---: | ---: | ---: |
| Reference | 0.1/1/5/10 cm | 10 cm | 15,096 | 1.75 s |
| Candidate | 0.5/1/5/10 cm | 20 cm | 6,360 | 0.42 s |

The candidate changed `k_eff` by only +5.52 pcm, but it failed the field
criteria: the maximum resolved region/group flux error was 35.3% in the
parked-control-rod region, and radial/axial power-shape RMS errors were 4.30%
and 3.07%. The required limits are 0.5% flux and 1% RMS power shape.
Consequently the 15,096-cell reference mesh remains the production mesh.

Power profiles are compared as line-power density, not raw per-bin power, so
different bin widths do not enter the error metric as a false shape change.

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

## CE Reference And SPH

The canonical JSON export now contains raw OpenMC tally means and standard
deviations for every resolved region and energy group. It also contains a
cylindrical `kappa-fission` mesh tally for radial and axial power comparison.
These are raw per-source-particle tally results in fast-to-thermal group order;
the independently normalized GeN-Foam auxiliary flux is not used for SPH.

For region `i` and group `g`, the OpenMC and diffusion region-integrated fluxes
are first normalized to equal total fission production. The direct fixed-point
target is

```text
mu_i,g(target) = Phi_i,g(CE) / Phi_i,g(diffusion)
```

The update is under-relaxed in log space with damping 0.5. For each source
group, `mu` multiplies absorption, fission, kappa-fission, and the full
scattering row. It also multiplies the transport cross section, which is
equivalent to

```text
D_i,g(SPH) = D_i,g / mu_i,g
```

`chi` is unchanged. The iteration stops after two consecutive iterations with
all statistically active flux errors below 0.2% and a `k_eff` change below
5 pcm. Factors are bounded and the fit fails explicitly on non-convergence.
They are not adjusted directly to force the eigenvalue.

`SphFactorSet` is immutable and records the factors, active mask, convergence
history, source fingerprint, mesh, algorithm version, and provisional status.
A factor file is rejected when its MGXS JSON, XML geometry, mesh, or algorithm
version no longer matches.

The tracked two- and four-group development exports contain the raw CE
region/group flux and cylindrical power payloads. Their CE `k_eff` uncertainty
is about 59 pcm, so they can exercise the SPH workflow but cannot qualify a
production factor set. Final qualification still requires a regenerated
reference with uncertainty at or below 15 pcm.

## Persistent Cache

`backend/openmc_to_diffusion.py --prepare-cache` prepares a reusable cache under
`backend/.cache/openmc_diffusion` by default. The fingerprint includes:

- canonical MGXS JSON contents;
- colocated `model.xml` contents;
- mesh-spacing configuration;
- the complete SPH factor artifact and algorithm version, when applied;
- solver tolerances and algorithm schema.

The cache persists:

- radial and axial mesh edges;
- cell-wise diffusion, absorption, fission, kappa-fission, spectrum, P0
  scattering, and resolved-region index arrays;
- one CSR spatial matrix per energy group;
- the converged clean-core eigenvalue and flux.

SuperLU factorization objects are process-local and are not serialized. A
backend process can load prepared CSR matrices and the clean result, then build
in-memory factorizations only when a new solve needs them. A missing or stale
cache is rebuilt synchronously. The cache is an optimization, not an alternate
source of nuclear data.

## Online Contract

The backend exposes the four-group clean-core model at:

```text
GET  /api/multigroup-diffusion/state
POST /api/multigroup-diffusion/recompute
```

Startup loads the persisted clean solution and corrected CSR operators when a
matching factor artifact exists. A fresh solve runs only on explicit
recompute. Missing factors produce a clearly marked uncorrected, provisional
result. Rod position is not an input to this service; immediate rod behavior
remains in point kinetics.

## Qualification Criteria

A final factor set is qualified only when:

- CE `k_eff` uncertainty is at most 15 pcm;
- corrected `k_eff` is within 50 pcm of the CE mean;
- every statistically active region/group flux is within 1% of CE;
- normalized radial and axial power profiles are within 2% RMS and 5%
  pointwise;
- repeated corrected solves agree within 1 pcm.

Until those conditions are met, the web page reports the result as
provisional rather than treating the calculation as a calibrated surrogate.

# Results multigroup diffusion w/o SPH
| Groups | MG OpenMC error vs CE | Diffusion error vs CE | Approx. diffusion bias vs MG |
| -----: | --------------------: | --------------------: | ---------------------------: |
|      1 |             +1908 pcm |             +1800 pcm |                     −108 pcm |
|      2 |              +806 pcm |             +2000 pcm |                    +1194 pcm |
|      4 |              +426 pcm |               −94 pcm |                     −520 pcm |
|      8 |              +166 pcm |              −366 pcm |                     −532 pcm |
|     16 |               +66 pcm |              −400 pcm |                     −466 pcm |

These values are the uncorrected baseline. They are retained to show the
group-condensation trend and are not SPH qualification results.

# Inspiration

On the use of the SPH method in nodal diffusion analyses of SFR cores - 10.1016/j.anucene.2015.06.007

- Regions for SPH
- Reactivity feedbacks
- Reflector boundary conditions
- Discontinuity factors