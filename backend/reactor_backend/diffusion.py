"""Reusable one-group diffusion solvers for annular reactor studies.

This module mirrors the numerical models developed in ``theory/reactorModel.ipynb``
so the notebook can stay focused on explanation and plotting while the actual
matrix assembly / eigenvalue solves live in importable Python code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class Region:
    name: str
    D: float
    Sigma_a: float
    nuSigma_f: float


@dataclass(frozen=True)
class AnnularModel:
    R_inner: float
    R_fuel: float
    R_refl: float
    mod_inner: Region
    fuel: Region
    reflector: Region
    extrap_factor: float = 2.13

    @property
    def R_extrap(self) -> float:
        return self.R_refl + self.extrap_factor * self.reflector.D

    def region_at(self, r: float) -> Region:
        if r < self.R_inner:
            return self.mod_inner
        if r < self.R_fuel:
            return self.fuel
        return self.reflector


@dataclass(frozen=True)
class Model2D:
    base: AnnularModel
    H: float
    H_refl: float
    r_rod: float
    dSa_rod: float
    extrap_factor: float = 2.13

    @property
    def H_extrap(self) -> float:
        return self.H + 2 * self.H_refl + 2 * self.extrap_factor * self.base.reflector.D

    @property
    def R_extrap(self) -> float:
        return self.base.R_extrap

    def region_at(self, r: float, z: float, x_insert: float = 0.0) -> Region:
        base = self.base
        half_height = self.H / 2
        if abs(z) > half_height:
            return base.reflector

        z_tip = half_height - x_insert * self.H
        if r < self.r_rod and z > z_tip:
            return Region(
                "rod",
                base.mod_inner.D,
                base.mod_inner.Sigma_a + self.dSa_rod,
                0.0,
            )

        return base.region_at(r)


def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b)


def solve_tridiagonal(lower, diagonal, upper, rhs):
    n = len(diagonal)
    c_prime = [0.0] * n
    d_prime = [0.0] * n
    c_prime[0] = upper[0] / diagonal[0] if n > 1 else 0.0
    d_prime[0] = rhs[0] / diagonal[0]

    for i in range(1, n):
        denom = diagonal[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    solution = [0.0] * n
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]

    return solution


def build_operators(model: AnnularModel, dr: float = 2.0):
    if dr <= 0.0:
        raise ValueError("dr must be positive for the 1D diffusion mesh")

    r_max = model.R_extrap
    cell_count = max(10, int(r_max / dr))
    radii = [(i + 0.5) * dr for i in range(cell_count)]
    lower = [0.0] * (cell_count - 1)
    diagonal = [0.0] * cell_count
    upper = [0.0] * (cell_count - 1)
    fiss = [0.0] * cell_count
    region_names: list[str] = []

    for i, r in enumerate(radii):
        region = model.region_at(r)
        region_names.append(region.name)
        fiss[i] = region.nuSigma_f * r

        r_left = max(r - 0.5 * dr, 0.0)
        r_right = r + 0.5 * dr

        if i == 0:
            leak_left = 0.0
        else:
            region_left = model.region_at(r - dr)
            d_left = harmonic_mean(region_left.D, region.D)
            leak_left = r_left * d_left / dr**2
            lower[i - 1] = -leak_left

        if i == cell_count - 1:
            leak_right = r_right * region.D / dr**2
        else:
            region_right = model.region_at(r + dr)
            d_right = harmonic_mean(region.D, region_right.D)
            leak_right = r_right * d_right / dr**2
            upper[i] = -leak_right

        diagonal[i] = leak_left + leak_right + region.Sigma_a * r

    return lower, diagonal, upper, fiss, radii, region_names


def solve_keff(model: AnnularModel, dr: float = 2.0, tol: float = 1e-8, max_iter: int = 600):
    lower, diagonal, upper, fiss, radii, region_names = build_operators(model, dr)
    if max(fiss, default=0.0) <= 0.0:
        raise ValueError("solve_keff requires at least one fissile region on the radial mesh")

    phi = [1.0] * len(radii)
    k = 1.0

    for iteration in range(1, max_iter + 1):
        rhs = [fission / k * flux for fission, flux in zip(fiss, phi)]
        phi_next = solve_tridiagonal(lower, diagonal, upper, rhs)
        prod_old = sum(fission * flux for fission, flux in zip(fiss, phi))
        if prod_old <= 0.0:
            raise RuntimeError("Power iteration encountered a zero fission source")

        prod_new = sum(fission * flux for fission, flux in zip(fiss, phi_next))
        if prod_new <= 0.0:
            raise RuntimeError("Power iteration collapsed to a zero fission source")

        k_next = k * prod_new / prod_old
        peak = max(phi_next)
        if peak <= 0.0:
            raise RuntimeError("Power iteration produced a non-positive flux shape")

        phi_norm = [value / peak for value in phi_next]

        if abs(k_next - k) < tol:
            return {
                "k_eff": k_next,
                "flux": phi_norm,
                "flux_raw": phi_next,
                "radii": radii,
                "regions": region_names,
                "iterations": iteration,
            }

        phi, k = phi_norm, k_next

    raise RuntimeError("Power iteration did not converge")


def with_fuel_radius(model: AnnularModel, r_fuel: float) -> AnnularModel:
    thickness = model.R_refl - model.R_fuel
    return replace(model, R_fuel=r_fuel, R_refl=r_fuel + thickness)


def find_critical_radius(
    model: AnnularModel,
    lo_cm: float,
    hi_cm: float,
    dr: float = 2.0,
    tol: float = 1e-5,
    max_it: int = 35,
):
    solution_lo = solve_keff(with_fuel_radius(model, lo_cm), dr)
    solution_hi = solve_keff(with_fuel_radius(model, hi_cm), dr)
    if solution_lo["k_eff"] > 1.0 or solution_hi["k_eff"] < 1.0:
        raise ValueError(
            f"k_eff not bracketed: k({lo_cm:.0f})={solution_lo['k_eff']:.4f}, "
            f"k({hi_cm:.0f})={solution_hi['k_eff']:.4f}"
        )

    lo, hi = lo_cm, hi_cm
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        trial_model = with_fuel_radius(model, mid)
        solution = solve_keff(trial_model, dr)
        if abs(solution["k_eff"] - 1.0) < tol:
            return trial_model, solution
        if solution["k_eff"] < 1.0:
            lo = mid
        else:
            hi = mid

    mid = 0.5 * (lo + hi)
    final_model = with_fuel_radius(model, mid)
    return final_model, solve_keff(final_model, dr)


def make_rod_region(inner_moderator: Region, delta_sigma_a: float, x_insert: float) -> Region:
    return Region(
        name="rod",
        D=inner_moderator.D,
        Sigma_a=inner_moderator.Sigma_a + x_insert * delta_sigma_a,
        nuSigma_f=0.0,
    )


def rodded_model(
    base: AnnularModel,
    x_insert: float,
    rod_radius: float,
    delta_sigma_a: float,
) -> AnnularModel:
    rod_region = make_rod_region(base.mod_inner, delta_sigma_a, x_insert)

    class RoddedModel(AnnularModel):
        def region_at(self, r: float) -> Region:
            if r < rod_radius:
                return rod_region
            return super().region_at(r)

    return RoddedModel(
        R_inner=base.R_inner,
        R_fuel=base.R_fuel,
        R_refl=base.R_refl,
        mod_inner=base.mod_inner,
        fuel=base.fuel,
        reflector=base.reflector,
        extrap_factor=base.extrap_factor,
    )


def build_2d_matrices(model: Model2D, dr: float, dz: float, x_insert: float = 0.0):
    if dr <= 0.0 or dz <= 0.0:
        raise ValueError("dr and dz must be positive for the 2D diffusion mesh")

    nr = int(round(model.R_extrap / dr))
    nz = int(round(model.H_extrap / dz))
    if nr < 1 or nz < 1:
        raise ValueError(
            "dr and dz are too coarse for the model extent; need at least one cell in each direction"
        )

    cell_count = nr * nz

    r_grid = (np.arange(nr) + 0.5) * dr
    z_grid = -model.H_extrap / 2 + (np.arange(nz) + 0.5) * dz
    r_face = np.arange(nr + 1) * dr

    r2d = r_grid[:, None]
    z2d = z_grid[None, :]
    base = model.base
    half_height = model.H / 2
    z_tip = half_height - x_insert * model.H

    axial_reflector = np.abs(z2d) > half_height
    rod = (r2d < model.r_rod) & (z2d > z_tip) & ~axial_reflector
    inner = (r2d < base.R_inner) & ~rod & ~axial_reflector
    fuel = (r2d >= base.R_inner) & (r2d < base.R_fuel) & ~axial_reflector

    rod_sigma_a = base.mod_inner.Sigma_a + model.dSa_rod

    def select(v_axial, v_rod, v_inner, v_fuel, v_reflector):
        return np.where(
            axial_reflector,
            v_axial,
            np.where(
                rod,
                v_rod,
                np.where(inner, v_inner, np.where(fuel, v_fuel, v_reflector)),
            ),
        )

    d_grid = select(
        base.reflector.D,
        base.mod_inner.D,
        base.mod_inner.D,
        base.fuel.D,
        base.reflector.D,
    )
    sigma_a_grid = select(
        base.reflector.Sigma_a,
        rod_sigma_a,
        base.mod_inner.Sigma_a,
        base.fuel.Sigma_a,
        base.reflector.Sigma_a,
    )
    nu_sigma_f_grid = np.where(fuel, base.fuel.nuSigma_f, 0.0)

    def harm(left, right):
        return 2 * left * right / (left + right + 1e-30)

    d_interface_r = harm(d_grid[:-1, :], d_grid[1:, :])
    d_r_left = np.zeros((nr, nz))
    d_r_right = np.zeros((nr, nz))
    d_r_left[1:, :] = d_interface_r
    d_r_right[:-1, :] = d_interface_r
    d_r_right[-1, :] = d_grid[-1, :]

    d_interface_z = harm(d_grid[:, :-1], d_grid[:, 1:])
    d_z_bottom = np.zeros((nr, nz))
    d_z_top = np.zeros((nr, nz))
    d_z_bottom[:, 1:] = d_interface_z
    d_z_bottom[:, 0] = d_grid[:, 0]
    d_z_top[:, :-1] = d_interface_z
    d_z_top[:, -1] = d_grid[:, -1]

    c_r_left = r_face[:-1, None] * d_r_left / (r2d * dr**2)
    c_r_right = r_face[1:, None] * d_r_right / (r2d * dr**2)
    c_z_bottom = d_z_bottom / dz**2
    c_z_top = d_z_top / dz**2

    diagonal = c_r_left + c_r_right + c_z_bottom + c_z_top + sigma_a_grid

    flat = np.arange(cell_count)
    ii = flat // nz
    jj = flat % nz

    rows = [flat]
    cols = [flat]
    vals = [diagonal.ravel()]

    mask = ii < nr - 1
    rows.append(flat[mask])
    cols.append(flat[mask] + nz)
    vals.append(-c_r_right.ravel()[mask])

    mask = ii > 0
    rows.append(flat[mask])
    cols.append(flat[mask] - nz)
    vals.append(-c_r_left.ravel()[mask])

    mask = jj < nz - 1
    rows.append(flat[mask])
    cols.append(flat[mask] + 1)
    vals.append(-c_z_top.ravel()[mask])

    mask = jj > 0
    rows.append(flat[mask])
    cols.append(flat[mask] - 1)
    vals.append(-c_z_bottom.ravel()[mask])

    loss = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(cell_count, cell_count),
    )
    fission = sp.diags(nu_sigma_f_grid.ravel(), format="csr")
    return loss, fission, r_grid, z_grid, nr, nz


def solve_2d(
    model: Model2D,
    dr: float = 3.0,
    dz: float = 3.0,
    x_insert: float = 0.0,
    phi0=None,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    loss, fission, r_grid, z_grid, nr, nz = build_2d_matrices(model, dr, dz, x_insert)
    cell_count = nr * nz
    if np.max(fission.diagonal()) <= 0.0:
        raise ValueError("solve_2d requires at least one fissile cell on the 2D mesh")

    if phi0 is None:
        phi = np.ones(cell_count)
    else:
        phi = np.abs(np.asarray(phi0, dtype=float).reshape(-1).copy())
        if phi.size != cell_count:
            raise ValueError(
                f"phi0 shape mismatch: expected {cell_count} entries, got {phi.size}"
            )

    phi_peak = phi.max()
    if phi_peak <= 0:
        raise ValueError("phi0 must contain at least one positive value")

    phi /= phi_peak

    solve_loss = spla.factorized(loss.tocsc())

    k = 1.0
    for iteration in range(max_iter):
        source = fission @ phi
        source_sum = source.sum()
        if source_sum <= 0.0:
            raise RuntimeError("2D power iteration encountered a zero fission source")

        phi_new = solve_loss(source)
        phi_new = np.maximum(phi_new, 0.0)
        phi_new_peak = phi_new.max()
        if phi_new_peak <= 0.0:
            raise RuntimeError("2D power iteration collapsed to a zero flux shape")

        source_new = fission @ phi_new
        source_new_sum = source_new.sum()
        if source_new_sum <= 0.0:
            raise RuntimeError("2D power iteration collapsed to a zero fission source")

        k_new = source_new_sum / source_sum
        phi = phi_new / phi_new_peak
        if iteration > 3 and abs(k_new - k) < tol:
            k = k_new
            break
        k = k_new

    return {
        "k_eff": k,
        "phi": phi.reshape(nr, nz),
        "r_grid": r_grid,
        "z_grid": z_grid,
        "Nr": nr,
        "Nz": nz,
        "iterations": iteration + 1,
    }


def rho_pcm(k: float) -> float:
    return (k - 1.0) / k * 1e5


def with_fuel_radius_2d(model: Model2D, r_fuel: float, h_to_r: float = 2.0) -> Model2D:
    """Return a copy of *model* with R_fuel (and H = h_to_r × R_fuel) updated.

    The radial and axial reflector thicknesses are kept fixed.
    """
    refl_thickness = model.base.R_refl - model.base.R_fuel
    new_base = replace(model.base, R_fuel=r_fuel, R_refl=r_fuel + refl_thickness)
    return replace(model, base=new_base, H=h_to_r * r_fuel)


def find_critical_radius_2d(
    template: Model2D,
    lo_cm: float,
    hi_cm: float,
    dr: float = 5.0,
    dz: float = 5.0,
    tol: float = 1e-4,
    max_it: int = 35,
    h_to_r: float = 2.0,
) -> tuple[Model2D, dict]:
    """Find the fuel outer radius that makes a 2D r–z core critical.

    Bisects over ``R_fuel`` in [lo_cm, hi_cm] while keeping R_inner, all
    region cross-sections, H_refl, and the rod geometry fixed.  The active
    height is updated as ``H = h_to_r × R_fuel``.

    Parameters
    ----------
    template:
        A Model2D whose fixed parameters (R_inner, region constants, H_refl,
        r_rod, dSa_rod) define the search space.  R_fuel and H are varied.
    lo_cm, hi_cm:
        Bracket for R_fuel.  k_eff must be sub-critical at lo_cm and
        super-critical at hi_cm.
    dr, dz:
        Mesh spacing (cm) for the search-phase solves.  Coarser is faster;
        use ~3 cm for a final re-solve after the search.
    tol:
        Convergence criterion on |k_eff − 1|.
    max_it:
        Maximum bisection iterations.
    h_to_r:
        Aspect ratio H = h_to_r × R_fuel (default 2.0).

    Returns
    -------
    (Model2D, dict)
        The critical Model2D and the corresponding solve_2d result dict.
    """

    def candidate(r_fuel: float) -> Model2D:
        return with_fuel_radius_2d(template, r_fuel, h_to_r=h_to_r)

    def k_at(r_fuel: float) -> tuple[float, dict]:
        sol = solve_2d(candidate(r_fuel), dr=dr, dz=dz, x_insert=0.0, max_iter=300, tol=1e-5)
        return sol["k_eff"], sol

    k_lo, _ = k_at(lo_cm)
    k_hi, _ = k_at(hi_cm)
    if k_lo >= 1.0 or k_hi <= 1.0:
        raise ValueError(
            f"k_eff not bracketed: k({lo_cm:.0f})={k_lo:.5f}, k({hi_cm:.0f})={k_hi:.5f}. "
            "Widen the search range or check material constants."
        )

    lo, hi = lo_cm, hi_cm
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        k_mid, sol_mid = k_at(mid)
        if abs(k_mid - 1.0) < tol:
            return candidate(mid), sol_mid
        if k_mid < 1.0:
            lo = mid
        else:
            hi = mid

    mid = 0.5 * (lo + hi)
    final_model = candidate(mid)
    _, final_sol = k_at(mid)
    return final_model, final_sol


__all__ = [
    "AnnularModel",
    "Model2D",
    "Region",
    "build_2d_matrices",
    "build_operators",
    "find_critical_radius",
    "find_critical_radius_2d",
    "harmonic_mean",
    "make_rod_region",
    "rho_pcm",
    "rodded_model",
    "solve_2d",
    "solve_keff",
    "solve_tridiagonal",
    "with_fuel_radius",
    "with_fuel_radius_2d",
]
