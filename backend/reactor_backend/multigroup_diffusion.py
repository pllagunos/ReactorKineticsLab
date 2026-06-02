"""Reusable multigroup diffusion solvers for layered cylindrical reactor models."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .diffusion import rho_pcm


def _as_1d(values, group_count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != group_count:
        raise ValueError(f"{name} must have length {group_count}, got {array.size}")
    return array


def _as_2d(values, group_count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (group_count, group_count):
        raise ValueError(
            f"{name} must have shape {(group_count, group_count)}, got {array.shape}"
        )
    return array


def _broadcast_to_groups(values, group_count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(group_count, float(array[0]), dtype=float)
    if array.size != group_count:
        raise ValueError(f"{name} must be scalar or length {group_count}, got {array.size}")
    return array.copy()


def _harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 2.0 * left * right / (left + right + 1e-30)


@dataclass(frozen=True)
class MultiGroupRegion:
    name: str
    diffusion: np.ndarray
    absorption: np.ndarray
    nu_fission: np.ndarray
    chi: np.ndarray
    scatter: np.ndarray

    def __post_init__(self):
        group_count = np.asarray(self.diffusion, dtype=float).reshape(-1).size
        object.__setattr__(self, "diffusion", _as_1d(self.diffusion, group_count, "diffusion"))
        object.__setattr__(self, "absorption", _as_1d(self.absorption, group_count, "absorption"))
        object.__setattr__(self, "nu_fission", _as_1d(self.nu_fission, group_count, "nu_fission"))
        object.__setattr__(self, "chi", _as_1d(self.chi, group_count, "chi"))
        object.__setattr__(self, "scatter", _as_2d(self.scatter, group_count, "scatter"))

    @property
    def group_count(self) -> int:
        return self.diffusion.size

    def sanitized(self, max_diffusion: float = 1e6) -> "MultiGroupRegion":
        diffusion = self.diffusion.copy()
        for group in range(self.group_count):
            needs_fix = (
                not np.isfinite(diffusion[group])
                or diffusion[group] <= 0.0
                or diffusion[group] > max_diffusion
            )
            if not needs_fix:
                continue

            inactive_group = (
                self.absorption[group] == 0.0
                and self.nu_fission[group] == 0.0
                and self.chi[group] == 0.0
                and np.all(self.scatter[group, :] == 0.0)
                and np.all(self.scatter[:, group] == 0.0)
            )
            if not inactive_group:
                raise ValueError(
                    f"Region {self.name!r} has an invalid diffusion coefficient in active group {group + 1}"
                )

            replacement = None
            for other_group in range(self.group_count):
                if other_group == group:
                    continue
                candidate = diffusion[other_group]
                if np.isfinite(candidate) and 0.0 < candidate <= max_diffusion:
                    replacement = float(candidate)
                    break
            if replacement is None:
                replacement = 1.0
            diffusion[group] = replacement

        return replace(self, diffusion=diffusion)

    def with_absorber(self, delta_absorption) -> "MultiGroupRegion":
        return replace(
            self,
            absorption=self.absorption + _broadcast_to_groups(
                delta_absorption,
                self.group_count,
                "delta_absorption",
            ),
            nu_fission=np.zeros(self.group_count, dtype=float),
        )


@dataclass(frozen=True)
class CylindricalRegionZone2D:
    region: MultiGroupRegion
    r_max: float
    z_min: float
    z_max: float
    r_min: float = 0.0

    def __post_init__(self):
        if self.r_min < 0.0:
            raise ValueError("Zone r_min must be non-negative")
        if self.r_max <= self.r_min:
            raise ValueError("Zone r_max must be greater than r_min")
        if self.z_max <= self.z_min:
            raise ValueError("Zone z_max must be greater than z_min")


@dataclass(frozen=True)
class CylindricalLayeredModel2D:
    core_radius: float
    moderator_radius: float
    reflector_radius: float
    core_height: float
    outer_height: float
    core: MultiGroupRegion
    moderator: MultiGroupRegion
    reflector: MultiGroupRegion
    rod_radius: float = 0.0
    delta_absorption_rod: float | np.ndarray = 0.0
    extrap_factor: float = 2.13
    zones: tuple[CylindricalRegionZone2D, ...] = ()

    def __post_init__(self):
        group_count = self.core.group_count
        if self.moderator.group_count != group_count or self.reflector.group_count != group_count:
            raise ValueError("All regions must use the same number of energy groups")
        for zone in self.zones:
            if zone.region.group_count != group_count:
                raise ValueError("All zones must use the same number of energy groups")

    @property
    def group_count(self) -> int:
        return self.core.group_count

    @property
    def max_boundary_diffusion(self) -> float:
        return float(np.max(self.reflector.diffusion))

    @property
    def R_extrap(self) -> float:
        return self.reflector_radius + self.extrap_factor * self.max_boundary_diffusion

    @property
    def H_extrap(self) -> float:
        return self.outer_height + 2.0 * self.extrap_factor * self.max_boundary_diffusion


def build_multigroup_2d_matrices(
    model: CylindricalLayeredModel2D,
    dr: float,
    dz: float,
    x_insert: float = 0.0,
):
    if dr <= 0.0 or dz <= 0.0:
        raise ValueError("dr and dz must be positive for the multigroup diffusion mesh")

    nr = int(round(model.R_extrap / dr))
    nz = int(round(model.H_extrap / dz))
    if nr < 1 or nz < 1:
        raise ValueError("dr and dz are too coarse for the model extent")

    group_count = model.group_count
    cell_count = nr * nz
    r_grid = (np.arange(nr) + 0.5) * dr
    z_grid = -model.H_extrap / 2.0 + (np.arange(nz) + 0.5) * dz
    r_face = np.arange(nr + 1) * dr

    r2d = r_grid[:, None]
    z2d = z_grid[None, :]
    outer_half_height = 0.5 * model.outer_height
    core_half_height = 0.5 * model.core_height

    rod_tip = core_half_height - x_insert * model.core_height
    rod_mask = (
        (r2d < model.rod_radius)
        & (z2d > rod_tip)
        & (np.abs(z2d) <= core_half_height)
    )

    diffusion = np.zeros((group_count, nr, nz), dtype=float)
    absorption = np.zeros((group_count, nr, nz), dtype=float)
    nu_fission = np.zeros((group_count, nr, nz), dtype=float)
    chi = np.zeros((group_count, nr, nz), dtype=float)
    scatter = np.zeros((group_count, group_count, nr, nz), dtype=float)

    zone_entries: list[tuple[np.ndarray, MultiGroupRegion]] = []
    if model.zones:
        for zone in model.zones:
            zone_mask = (
                (r2d >= zone.r_min)
                & (r2d < zone.r_max)
                & (z2d >= zone.z_min)
                & (z2d < zone.z_max)
            )
            zone_entries.append((zone_mask, zone.region))
    else:
        core_mask = (r2d < model.core_radius) & (np.abs(z2d) <= core_half_height)
        moderator_mask = (r2d < model.moderator_radius) & ~core_mask & (np.abs(z2d) <= outer_half_height)
        reflector_mask = ~core_mask & ~moderator_mask
        zone_entries.extend([
            (reflector_mask, model.reflector),
            (moderator_mask, model.moderator),
            (core_mask, model.core),
        ])

    for group in range(group_count):
        diffusion[group].fill(model.reflector.diffusion[group])
        absorption[group].fill(model.reflector.absorption[group])
        nu_fission[group].fill(model.reflector.nu_fission[group])
        chi[group].fill(model.reflector.chi[group])
        for zone_mask, zone_region in zone_entries:
            diffusion[group] = np.where(zone_mask, zone_region.diffusion[group], diffusion[group])
            absorption[group] = np.where(zone_mask, zone_region.absorption[group], absorption[group])
            nu_fission[group] = np.where(zone_mask, zone_region.nu_fission[group], nu_fission[group])
            chi[group] = np.where(zone_mask, zone_region.chi[group], chi[group])

        absorption[group] = np.where(
            rod_mask,
            absorption[group]
            + _broadcast_to_groups(model.delta_absorption_rod, group_count, "delta_absorption_rod")[group],
            absorption[group],
        )
        nu_fission[group] = np.where(rod_mask, 0.0, nu_fission[group])
        chi[group] = np.where(rod_mask, 0.0, chi[group])

        for destination in range(group_count):
            scatter[group, destination].fill(model.reflector.scatter[group, destination])
            for zone_mask, zone_region in zone_entries:
                scatter[group, destination] = np.where(
                    zone_mask,
                    zone_region.scatter[group, destination],
                    scatter[group, destination],
                )
            scatter[group, destination] = np.where(
                rod_mask,
                0.0,
                scatter[group, destination],
            )

    flat = np.arange(cell_count)
    ii = flat // nz
    jj = flat % nz
    blocks = [[None for _ in range(group_count)] for _ in range(group_count)]
    fission_blocks = [[None for _ in range(group_count)] for _ in range(group_count)]

    for group in range(group_count):
        d_grid = diffusion[group]
        d_interface_r = _harmonic_mean(d_grid[:-1, :], d_grid[1:, :])
        d_r_left = np.zeros((nr, nz), dtype=float)
        d_r_right = np.zeros((nr, nz), dtype=float)
        d_r_left[1:, :] = d_interface_r
        d_r_right[:-1, :] = d_interface_r
        d_r_right[-1, :] = d_grid[-1, :]

        d_interface_z = _harmonic_mean(d_grid[:, :-1], d_grid[:, 1:])
        d_z_bottom = np.zeros((nr, nz), dtype=float)
        d_z_top = np.zeros((nr, nz), dtype=float)
        d_z_bottom[:, 1:] = d_interface_z
        d_z_bottom[:, 0] = d_grid[:, 0]
        d_z_top[:, :-1] = d_interface_z
        d_z_top[:, -1] = d_grid[:, -1]

        c_r_left = r_face[:-1, None] * d_r_left / (r2d * dr**2)
        c_r_right = r_face[1:, None] * d_r_right / (r2d * dr**2)
        c_z_bottom = d_z_bottom / dz**2
        c_z_top = d_z_top / dz**2

        removal = absorption[group].copy()
        for destination in range(group_count):
            if destination != group:
                removal += scatter[group, destination]

        diagonal = c_r_left + c_r_right + c_z_bottom + c_z_top + removal

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

        blocks[group][group] = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(cell_count, cell_count),
        )

        for source in range(group_count):
            if source == group:
                continue
            blocks[group][source] = sp.diags(-scatter[source, group].ravel(), format="csr")

        for source in range(group_count):
            fission_blocks[group][source] = sp.diags(
                (chi[group] * nu_fission[source]).ravel(),
                format="csr",
            )

    loss = sp.bmat(blocks, format="csr")
    fission = sp.bmat(fission_blocks, format="csr")
    return loss, fission, r_grid, z_grid, nr, nz, group_count


def solve_multigroup_2d(
    model: CylindricalLayeredModel2D,
    dr: float = 3.0,
    dz: float = 3.0,
    x_insert: float = 0.0,
    phi0=None,
    max_iter: int = 200,
    tol: float = 1e-6,
):
    loss, fission, r_grid, z_grid, nr, nz, group_count = build_multigroup_2d_matrices(
        model,
        dr,
        dz,
        x_insert,
    )
    cell_count = nr * nz
    vector_size = group_count * cell_count
    if np.max(fission.diagonal()) <= 0.0:
        raise ValueError("solve_multigroup_2d requires at least one fissile cell on the mesh")

    if phi0 is None:
        phi = np.ones(vector_size, dtype=float)
    else:
        phi = np.abs(np.asarray(phi0, dtype=float).reshape(-1).copy())
        if phi.size != vector_size:
            raise ValueError(
                f"phi0 shape mismatch: expected {vector_size} entries, got {phi.size}"
            )

    phi_peak = phi.max()
    if phi_peak <= 0.0:
        raise ValueError("phi0 must contain at least one positive value")

    phi /= phi_peak
    solve_loss = spla.factorized(loss.tocsc())

    k_eff = 1.0
    for iteration in range(1, max_iter + 1):
        source = fission @ phi
        source_sum = source.sum()
        if source_sum <= 0.0:
            raise RuntimeError("Multigroup power iteration encountered a zero fission source")

        phi_new = np.maximum(solve_loss(source), 0.0)
        phi_new_peak = phi_new.max()
        if phi_new_peak <= 0.0:
            raise RuntimeError("Multigroup power iteration collapsed to a zero flux shape")

        source_new = fission @ phi_new
        source_new_sum = source_new.sum()
        if source_new_sum <= 0.0:
            raise RuntimeError("Multigroup power iteration collapsed to a zero fission source")

        k_new = source_new_sum / source_sum
        phi = phi_new / phi_new_peak
        if iteration > 3 and abs(k_new - k_eff) < tol:
            k_eff = k_new
            break
        k_eff = k_new

    phi_groups = phi.reshape(group_count, nr, nz)
    return {
        "k_eff": float(k_eff),
        "phi": phi_groups.sum(axis=0),
        "phi_groups": phi_groups,
        "r_grid": r_grid,
        "z_grid": z_grid,
        "Nr": nr,
        "Nz": nz,
        "group_count": group_count,
        "iterations": iteration,
    }


def scan_multigroup_rod_worth_2d(
    model: CylindricalLayeredModel2D,
    x_values: np.ndarray | list[float] | None = None,
    dr: float = 3.0,
    dz: float = 3.0,
    max_iter: int = 300,
    tol: float = 1e-6,
    warm_start: bool = True,
):
    if x_values is None:
        x_arr = np.linspace(0.0, 1.0, 11)
    else:
        x_arr = np.asarray(x_values, dtype=float).reshape(-1)
    if x_arr.size == 0:
        raise ValueError("x_values must not be empty")
    if np.any((x_arr < 0.0) | (x_arr > 1.0)):
        raise ValueError("x_values must lie in [0, 1]")

    clean_solution = solve_multigroup_2d(
        model,
        dr=dr,
        dz=dz,
        x_insert=0.0,
        max_iter=max_iter,
        tol=tol,
    )
    rho_clean_pcm = rho_pcm(clean_solution["k_eff"])
    phi_guess = clean_solution["phi_groups"] if warm_start else None

    k_values = np.zeros(x_arr.size, dtype=float)
    rho_total_pcm = np.zeros(x_arr.size, dtype=float)
    iterations: list[int] = []

    for index, x_insert in enumerate(x_arr):
        solution = solve_multigroup_2d(
            model,
            dr=dr,
            dz=dz,
            x_insert=float(x_insert),
            phi0=phi_guess,
            max_iter=max_iter,
            tol=tol,
        )
        k_values[index] = solution["k_eff"]
        rho_total_pcm[index] = rho_pcm(solution["k_eff"])
        iterations.append(solution["iterations"])
        if warm_start:
            phi_guess = solution["phi_groups"]

    delta_rho_pcm = rho_total_pcm - rho_clean_pcm
    critical_insertion = None
    for index in range(x_arr.size - 1):
        rho0 = rho_total_pcm[index]
        rho1 = rho_total_pcm[index + 1]
        if rho0 == 0.0:
            critical_insertion = float(x_arr[index])
            break
        if rho0 * rho1 <= 0.0 and rho1 != rho0:
            x0 = x_arr[index]
            x1 = x_arr[index + 1]
            critical_insertion = float(x0 + (0.0 - rho0) * (x1 - x0) / (rho1 - rho0))
            break
    if critical_insertion is None and rho_total_pcm[-1] == 0.0:
        critical_insertion = float(x_arr[-1])

    return {
        "x_insert": x_arr,
        "k_eff": k_values,
        "rho_total_pcm": rho_total_pcm,
        "delta_rho_pcm": delta_rho_pcm,
        "rho_clean_pcm": rho_clean_pcm,
        "critical_insertion_fraction": critical_insertion,
        "iterations": iterations,
    }