"""Reusable multigroup diffusion solvers for layered cylindrical reactor models."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

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


@dataclass(frozen=True)
class ConcentricMeshSpacing:
    fuel_radial_cm: float = 0.1
    core_coolant_radial_cm: float = 1.0
    moderator_radial_cm: float = 5.0
    reflector_radial_cm: float = 10.0
    axial_cm: float = 10.0

    def __post_init__(self):
        for name, value in self.as_dict().items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive spacing")

    def as_dict(self) -> dict[str, float]:
        return {
            "fuel_radial_cm": float(self.fuel_radial_cm),
            "core_coolant_radial_cm": float(self.core_coolant_radial_cm),
            "moderator_radial_cm": float(self.moderator_radial_cm),
            "reflector_radial_cm": float(self.reflector_radial_cm),
            "axial_cm": float(self.axial_cm),
        }


@dataclass(frozen=True)
class CylindricalMesh2D:
    r_edges: np.ndarray
    z_edges: np.ndarray

    def __post_init__(self):
        r_edges = np.asarray(self.r_edges, dtype=float).reshape(-1)
        z_edges = np.asarray(self.z_edges, dtype=float).reshape(-1)
        if r_edges.size < 2 or z_edges.size < 2:
            raise ValueError("A cylindrical mesh requires at least one cell")
        if r_edges[0] != 0.0 or np.any(np.diff(r_edges) <= 0.0):
            raise ValueError("r_edges must start at zero and increase strictly")
        if np.any(np.diff(z_edges) <= 0.0):
            raise ValueError("z_edges must increase strictly")
        object.__setattr__(self, "r_edges", r_edges)
        object.__setattr__(self, "z_edges", z_edges)

    @property
    def r_grid(self) -> np.ndarray:
        return 0.5 * (self.r_edges[:-1] + self.r_edges[1:])

    @property
    def z_grid(self) -> np.ndarray:
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def nr(self) -> int:
        return self.r_edges.size - 1

    @property
    def nz(self) -> int:
        return self.z_edges.size - 1

    @property
    def cell_count(self) -> int:
        return self.nr * self.nz

    @property
    def volumes(self) -> np.ndarray:
        annular_areas = np.pi * (
            self.r_edges[1:] ** 2 - self.r_edges[:-1] ** 2
        )
        return (annular_areas[:, None] * np.diff(self.z_edges)[None, :]).ravel()


@dataclass(frozen=True)
class MultiGroupDiffusionSystem:
    model: CylindricalLayeredModel2D
    mesh: CylindricalMesh2D
    operators: tuple[sp.csr_matrix, ...]
    diffusion: np.ndarray
    absorption: np.ndarray
    nu_fission: np.ndarray
    chi: np.ndarray
    scatter: np.ndarray
    x_insert: float

    @property
    def group_count(self) -> int:
        return self.model.group_count

    @property
    def cell_count(self) -> int:
        return self.mesh.cell_count


def _subdivide_interval(start: float, stop: float, target: float) -> np.ndarray:
    count = max(1, int(np.ceil((stop - start) / target - 1.0e-12)))
    return np.linspace(start, stop, count + 1)


def _fuel_radial_intervals(
    model: CylindricalLayeredModel2D,
) -> tuple[tuple[float, float], ...]:
    intervals = [
        (zone.r_min, zone.r_max)
        for zone in model.zones
        if zone.region.name.startswith("core_fuel_ring_")
    ]
    return tuple(intervals)


def build_concentric_mesh(
    model: CylindricalLayeredModel2D,
    spacing: ConcentricMeshSpacing = ConcentricMeshSpacing(),
) -> CylindricalMesh2D:
    radial_boundaries = {
        0.0,
        model.core_radius,
        model.moderator_radius,
        model.reflector_radius,
        model.R_extrap,
    }
    axial_boundaries = {
        -0.5 * model.H_extrap,
        0.5 * model.H_extrap,
        -0.5 * model.outer_height,
        0.5 * model.outer_height,
        -0.5 * model.core_height,
        0.5 * model.core_height,
    }
    for zone in model.zones:
        radial_boundaries.update((zone.r_min, zone.r_max))
        axial_boundaries.update((zone.z_min, zone.z_max))

    radial = sorted(
        value
        for value in radial_boundaries
        if 0.0 <= value <= model.R_extrap
    )
    axial = sorted(
        value
        for value in axial_boundaries
        if -0.5 * model.H_extrap <= value <= 0.5 * model.H_extrap
    )
    fuel_intervals = _fuel_radial_intervals(model)

    radial_parts: list[np.ndarray] = []
    for index, (start, stop) in enumerate(zip(radial, radial[1:])):
        midpoint = 0.5 * (start + stop)
        if any(lower <= midpoint <= upper for lower, upper in fuel_intervals):
            target = spacing.fuel_radial_cm
        elif midpoint <= model.core_radius:
            target = spacing.core_coolant_radial_cm
        elif midpoint <= model.moderator_radius:
            target = spacing.moderator_radial_cm
        else:
            target = spacing.reflector_radial_cm
        part = _subdivide_interval(start, stop, target)
        radial_parts.append(part if index == 0 else part[1:])

    axial_parts: list[np.ndarray] = []
    for index, (start, stop) in enumerate(zip(axial, axial[1:])):
        part = _subdivide_interval(start, stop, spacing.axial_cm)
        axial_parts.append(part if index == 0 else part[1:])

    return CylindricalMesh2D(
        r_edges=np.concatenate(radial_parts),
        z_edges=np.concatenate(axial_parts),
    )


def _legacy_spacing(dr: float, dz: float) -> ConcentricMeshSpacing:
    if dr <= 0.0 or dz <= 0.0:
        raise ValueError("dr and dz must be positive for the multigroup diffusion mesh")
    return ConcentricMeshSpacing(
        fuel_radial_cm=dr,
        core_coolant_radial_cm=dr,
        moderator_radial_cm=dr,
        reflector_radial_cm=dr,
        axial_cm=dz,
    )


def _material_arrays(
    model: CylindricalLayeredModel2D,
    mesh: CylindricalMesh2D,
    x_insert: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= x_insert <= 1.0:
        raise ValueError("x_insert must lie in [0, 1]")

    group_count = model.group_count
    nr, nz = mesh.nr, mesh.nz
    r2d = mesh.r_grid[:, None]
    z2d = mesh.z_grid[None, :]
    core_half_height = 0.5 * model.core_height
    outer_half_height = 0.5 * model.outer_height

    zone_entries: list[tuple[np.ndarray, MultiGroupRegion]] = []
    if model.zones:
        for zone in model.zones:
            zone_entries.append(
                (
                    (r2d >= zone.r_min)
                    & (r2d < zone.r_max)
                    & (z2d >= zone.z_min)
                    & (z2d < zone.z_max),
                    zone.region,
                )
            )
    else:
        core_mask = (r2d < model.core_radius) & (
            np.abs(z2d) <= core_half_height
        )
        moderator_mask = (
            (r2d < model.moderator_radius)
            & ~core_mask
            & (np.abs(z2d) <= outer_half_height)
        )
        zone_entries.extend(
            [
                (~core_mask & ~moderator_mask, model.reflector),
                (moderator_mask, model.moderator),
                (core_mask, model.core),
            ]
        )

    diffusion = np.broadcast_to(
        model.reflector.diffusion, (nr, nz, group_count)
    ).copy()
    absorption = np.broadcast_to(
        model.reflector.absorption, (nr, nz, group_count)
    ).copy()
    nu_fission = np.broadcast_to(
        model.reflector.nu_fission, (nr, nz, group_count)
    ).copy()
    chi = np.broadcast_to(model.reflector.chi, (nr, nz, group_count)).copy()
    scatter = np.broadcast_to(
        model.reflector.scatter, (nr, nz, group_count, group_count)
    ).copy()

    for mask, region in zone_entries:
        diffusion[mask] = region.diffusion
        absorption[mask] = region.absorption
        nu_fission[mask] = region.nu_fission
        chi[mask] = region.chi
        scatter[mask] = region.scatter

    rod_tip = core_half_height - x_insert * model.core_height
    rod_mask = (
        (r2d < model.rod_radius)
        & (z2d > rod_tip)
        & (np.abs(z2d) <= core_half_height)
    )
    if np.any(rod_mask):
        absorption[rod_mask] += _broadcast_to_groups(
            model.delta_absorption_rod,
            group_count,
            "delta_absorption_rod",
        )
        nu_fission[rod_mask] = 0.0
        chi[rod_mask] = 0.0
        scatter[rod_mask] = 0.0

    return tuple(
        values.reshape((-1,) + values.shape[2:])
        for values in (diffusion, absorption, nu_fission, chi, scatter)
    )


def _spatial_operator(
    mesh: CylindricalMesh2D,
    diffusion: np.ndarray,
    removal: np.ndarray,
) -> sp.csr_matrix:
    nr, nz = mesh.nr, mesh.nz
    d_grid = diffusion.reshape(nr, nz)
    removal_grid = removal.reshape(nr, nz)
    dr = np.diff(mesh.r_edges)
    dz = np.diff(mesh.z_edges)
    radial_volume = 0.5 * (
        mesh.r_edges[1:] ** 2 - mesh.r_edges[:-1] ** 2
    )

    c_r_left = np.zeros((nr, nz), dtype=float)
    c_r_right = np.zeros((nr, nz), dtype=float)
    if nr > 1:
        conductance = 1.0 / (
            dr[:-1, None] / (2.0 * d_grid[:-1])
            + dr[1:, None] / (2.0 * d_grid[1:])
        )
        c_r_right[:-1] = (
            mesh.r_edges[1:-1, None]
            * conductance
            / radial_volume[:-1, None]
        )
        c_r_left[1:] = (
            mesh.r_edges[1:-1, None]
            * conductance
            / radial_volume[1:, None]
        )
    c_r_right[-1] = (
        mesh.r_edges[-1]
        * d_grid[-1]
        / (radial_volume[-1] * (0.5 * dr[-1]))
    )

    c_z_bottom = np.zeros((nr, nz), dtype=float)
    c_z_top = np.zeros((nr, nz), dtype=float)
    if nz > 1:
        conductance = 1.0 / (
            dz[:-1][None, :] / (2.0 * d_grid[:, :-1])
            + dz[1:][None, :] / (2.0 * d_grid[:, 1:])
        )
        c_z_top[:, :-1] = conductance / dz[:-1][None, :]
        c_z_bottom[:, 1:] = conductance / dz[1:][None, :]
    c_z_bottom[:, 0] = d_grid[:, 0] / (dz[0] * (0.5 * dz[0]))
    c_z_top[:, -1] = d_grid[:, -1] / (dz[-1] * (0.5 * dz[-1]))

    diagonal = (
        c_r_left
        + c_r_right
        + c_z_bottom
        + c_z_top
        + removal_grid
    )
    flat = np.arange(mesh.cell_count)
    ii = flat // nz
    jj = flat % nz
    rows = [flat]
    cols = [flat]
    values = [diagonal.ravel()]

    mask = ii < nr - 1
    rows.append(flat[mask])
    cols.append(flat[mask] + nz)
    values.append(-c_r_right.ravel()[mask])

    mask = ii > 0
    rows.append(flat[mask])
    cols.append(flat[mask] - nz)
    values.append(-c_r_left.ravel()[mask])

    mask = jj < nz - 1
    rows.append(flat[mask])
    cols.append(flat[mask] + 1)
    values.append(-c_z_top.ravel()[mask])

    mask = jj > 0
    rows.append(flat[mask])
    cols.append(flat[mask] - 1)
    values.append(-c_z_bottom.ravel()[mask])

    return sp.csr_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(mesh.cell_count, mesh.cell_count),
    )


def build_multigroup_2d_system(
    model: CylindricalLayeredModel2D,
    *,
    mesh: CylindricalMesh2D | None = None,
    spacing: ConcentricMeshSpacing = ConcentricMeshSpacing(),
    x_insert: float = 0.0,
) -> MultiGroupDiffusionSystem:
    mesh = mesh or build_concentric_mesh(model, spacing)
    diffusion, absorption, nu_fission, chi, scatter = _material_arrays(
        model, mesh, x_insert
    )
    operators = []
    for group in range(model.group_count):
        outscatter = np.sum(scatter[:, group, :], axis=1) - scatter[:, group, group]
        operators.append(
            _spatial_operator(
                mesh,
                diffusion[:, group],
                absorption[:, group] + outscatter,
            )
        )
    return MultiGroupDiffusionSystem(
        model=model,
        mesh=mesh,
        operators=tuple(operators),
        diffusion=diffusion,
        absorption=absorption,
        nu_fission=nu_fission,
        chi=chi,
        scatter=scatter,
        x_insert=float(x_insert),
    )


def _assemble_global_matrices(
    system: MultiGroupDiffusionSystem,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    group_count = system.group_count
    blocks: list[list[sp.spmatrix | None]] = [
        [None for _ in range(group_count)] for _ in range(group_count)
    ]
    fission_blocks: list[list[sp.spmatrix | None]] = [
        [None for _ in range(group_count)] for _ in range(group_count)
    ]
    for destination in range(group_count):
        for source in range(group_count):
            if source == destination:
                blocks[destination][source] = system.operators[destination]
            else:
                blocks[destination][source] = sp.diags(
                    -system.scatter[:, source, destination],
                    format="csr",
                )
            fission_blocks[destination][source] = sp.diags(
                system.chi[:, destination] * system.nu_fission[:, source],
                format="csr",
            )
    return (
        sp.bmat(blocks, format="csr"),
        sp.bmat(fission_blocks, format="csr"),
    )


def build_multigroup_2d_matrices(
    model: CylindricalLayeredModel2D,
    dr: float,
    dz: float,
    x_insert: float = 0.0,
):
    """Build the monolithic reference matrices for small verification cases."""
    system = build_multigroup_2d_system(
        model,
        spacing=_legacy_spacing(dr, dz),
        x_insert=x_insert,
    )
    loss, fission = _assemble_global_matrices(system)
    return (
        loss,
        fission,
        system.mesh.r_grid,
        system.mesh.z_grid,
        system.mesh.nr,
        system.mesh.nz,
        system.group_count,
    )


def _initial_flux(
    system: MultiGroupDiffusionSystem,
    phi0: Any,
) -> np.ndarray:
    expected_size = system.cell_count * system.group_count
    if phi0 is None:
        phi = np.ones((system.cell_count, system.group_count), dtype=float)
    else:
        supplied = np.abs(np.asarray(phi0, dtype=float))
        if supplied.size != expected_size:
            raise ValueError(
                f"phi0 shape mismatch: expected {expected_size} entries, "
                f"got {supplied.size}"
            )
        if supplied.shape == (
            system.group_count,
            system.mesh.nr,
            system.mesh.nz,
        ):
            phi = supplied.reshape(system.group_count, -1).T.copy()
        elif supplied.shape == (
            system.mesh.nr,
            system.mesh.nz,
            system.group_count,
        ):
            phi = supplied.reshape(-1, system.group_count).copy()
        else:
            phi = supplied.reshape(system.group_count, system.cell_count).T.copy()
    if not np.all(np.isfinite(phi)) or np.max(phi) <= 0.0:
        raise ValueError("phi0 must contain finite values and at least one positive value")
    return phi


def _fission_density(
    system: MultiGroupDiffusionSystem,
    phi: np.ndarray,
) -> np.ndarray:
    return np.sum(system.nu_fission * phi, axis=1)


def _fission_production(
    system: MultiGroupDiffusionSystem,
    phi: np.ndarray,
) -> float:
    return float(np.dot(system.mesh.volumes, _fission_density(system, phi)))


def _normalize_fission(
    system: MultiGroupDiffusionSystem,
    phi: np.ndarray,
) -> tuple[np.ndarray, float]:
    production = _fission_production(system, phi)
    if not np.isfinite(production) or production <= 0.0:
        raise RuntimeError("Multigroup iteration encountered a zero fission source")
    return phi / production, production


def _solution_payload(
    system: MultiGroupDiffusionSystem,
    phi: np.ndarray,
    *,
    k_eff: float,
    iterations: int,
    inner_iterations: list[int],
    converged: bool,
    timings: dict[str, float],
) -> dict[str, Any]:
    phi_groups = phi.T.reshape(
        system.group_count, system.mesh.nr, system.mesh.nz
    )
    scatter = system.scatter.copy()
    diagonal = np.arange(system.group_count)
    scatter[:, diagonal, diagonal] = 0.0
    fission_density = _fission_density(system, phi)
    incoming = np.einsum("csg,cs->cg", scatter, phi, optimize=True)
    loss = np.column_stack(
        [
            system.operators[group] @ phi[:, group]
            for group in range(system.group_count)
        ]
    )
    source = system.chi * fission_density[:, None] / k_eff + incoming
    weights = system.mesh.volumes[:, None]
    balance_residual = float(
        np.sum(weights * np.abs(loss - source))
        / max(np.sum(weights * np.abs(source)), 1.0e-30)
    )
    return {
        "k_eff": float(k_eff),
        "phi": phi_groups.sum(axis=0),
        "phi_groups": phi_groups,
        "r_grid": system.mesh.r_grid,
        "z_grid": system.mesh.z_grid,
        "r_edges": system.mesh.r_edges,
        "z_edges": system.mesh.z_edges,
        "Nr": system.mesh.nr,
        "Nz": system.mesh.nz,
        "cell_count": system.cell_count,
        "group_count": system.group_count,
        "iterations": iterations,
        "inner_iterations": inner_iterations,
        "converged": converged,
        "balance_residual": balance_residual,
        "timings_s": timings,
    }


def _solve_scattering_source(
    system: MultiGroupDiffusionSystem,
    group_solves: tuple[Any, ...],
    scatter: np.ndarray,
    fixed_fission_source: np.ndarray,
    phi_guess: np.ndarray,
    *,
    max_inner_iter: int,
    inner_tol: float,
) -> tuple[np.ndarray, int]:
    phi_inner = phi_guess.copy()
    gauss_seidel_sweeps = min(5, max_inner_iter)
    for sweep in range(1, gauss_seidel_sweeps + 1):
        before_sweep = phi_inner.copy()
        for destination in range(system.group_count):
            scatter_source = np.einsum(
                "cs,cs->c",
                scatter[:, :, destination],
                phi_inner,
                optimize=True,
            )
            rhs = fixed_fission_source[:, destination] + scatter_source
            solved = np.asarray(group_solves[destination](rhs))
            if not np.all(np.isfinite(solved)):
                raise RuntimeError(
                    f"Group {destination + 1} solve produced non-finite flux"
                )
            phi_inner[:, destination] = np.maximum(solved, 0.0)

        scale = max(float(np.max(phi_inner)), 1.0e-30)
        inner_error = float(np.max(np.abs(phi_inner - before_sweep)) / scale)
        if inner_error < inner_tol:
            return phi_inner, sweep

    remaining_iterations = max_inner_iter - gauss_seidel_sweeps
    if remaining_iterations <= 0:
        raise RuntimeError(
            f"Multigroup scattering iteration did not converge in "
            f"{max_inner_iter} sweeps"
        )

    vector_size = system.cell_count * system.group_count

    def matvec(vector: np.ndarray) -> np.ndarray:
        flux = vector.reshape(system.group_count, system.cell_count).T
        incoming = np.einsum("csg,cs->cg", scatter, flux, optimize=True)
        result = np.empty_like(flux)
        for group in range(system.group_count):
            result[:, group] = (
                system.operators[group] @ flux[:, group] - incoming[:, group]
            )
        return result.T.reshape(-1)

    def precondition(vector: np.ndarray) -> np.ndarray:
        source = vector.reshape(system.group_count, system.cell_count).T
        result = np.empty_like(source)
        for group in range(system.group_count):
            result[:, group] = group_solves[group](source[:, group])
        return result.T.reshape(-1)

    operator = spla.LinearOperator(
        (vector_size, vector_size),
        matvec=matvec,
        dtype=float,
    )
    preconditioner = spla.LinearOperator(
        (vector_size, vector_size),
        matvec=precondition,
        dtype=float,
    )
    krylov_iterations = 0

    def count_iteration(_residual: float) -> None:
        nonlocal krylov_iterations
        krylov_iterations += 1

    restart = min(30, remaining_iterations)
    restart_cycles = max(1, int(np.ceil(remaining_iterations / restart)))
    solved, info = spla.gmres(
        operator,
        fixed_fission_source.T.reshape(-1),
        x0=phi_inner.T.reshape(-1),
        rtol=inner_tol,
        atol=0.0,
        restart=restart,
        maxiter=restart_cycles,
        M=preconditioner,
        callback=count_iteration,
        callback_type="pr_norm",
    )
    if info != 0:
        raise RuntimeError(
            "Matrix-free accelerated scattering solve did not converge "
            f"within {remaining_iterations} Krylov iterations (info={info})"
        )
    phi_inner = solved.reshape(system.group_count, system.cell_count).T
    if not np.all(np.isfinite(phi_inner)):
        raise RuntimeError("Accelerated scattering solve produced non-finite flux")
    return np.maximum(phi_inner, 0.0), gauss_seidel_sweeps + krylov_iterations


def solve_multigroup_system(
    system: MultiGroupDiffusionSystem,
    *,
    phi0: Any = None,
    max_iter: int = 200,
    tol: float = 1.0e-6,
    source_tol: float = 1.0e-3,
    max_inner_iter: int = 100,
    inner_tol: float = 1.0e-4,
) -> dict[str, Any]:
    if max_iter < 1 or max_inner_iter < 1:
        raise ValueError("Iteration limits must be positive")
    if tol <= 0.0 or source_tol <= 0.0 or inner_tol <= 0.0:
        raise ValueError("Iteration tolerances must be positive")

    started = time.perf_counter()
    phi, _ = _normalize_fission(system, _initial_flux(system, phi0))
    factorization_started = time.perf_counter()
    group_solves = tuple(
        spla.factorized(operator.tocsc()) for operator in system.operators
    )
    factorization_seconds = time.perf_counter() - factorization_started

    scatter = system.scatter.copy()
    diagonal = np.arange(system.group_count)
    scatter[:, diagonal, diagonal] = 0.0
    volumes = system.mesh.volumes
    k_eff = 1.0
    converged = False
    inner_counts: list[int] = []

    for iteration in range(1, max_iter + 1):
        old_fission = _fission_density(system, phi)
        fixed_fission_source = system.chi * old_fission[:, None] / k_eff
        if iteration <= 5:
            scheduled_inner_tol = 1.0e-2
        elif iteration <= 20:
            scheduled_inner_tol = 1.0e-3
        elif iteration <= 35:
            scheduled_inner_tol = 5.0e-4
        else:
            scheduled_inner_tol = inner_tol
        effective_inner_tol = max(inner_tol, scheduled_inner_tol)
        phi_inner, inner_iteration = _solve_scattering_source(
            system,
            group_solves,
            scatter,
            fixed_fission_source,
            phi,
            max_inner_iter=max_inner_iter,
            inner_tol=effective_inner_tol,
        )
        inner_counts.append(inner_iteration)

        phi_new, production = _normalize_fission(system, phi_inner)
        k_new = k_eff * production
        new_fission = _fission_density(system, phi_new)
        source_error = float(
            np.dot(volumes, np.abs(new_fission - old_fission))
        )
        k_error = abs(k_new - k_eff) / max(abs(k_new), 1.0e-30)
        phi = phi_new
        k_eff = k_new
        final_inner_tolerance_active = effective_inner_tol <= inner_tol
        if (
            iteration > 1
            and final_inner_tolerance_active
            and k_error < tol
            and source_error < source_tol
        ):
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"Multigroup power iteration did not converge in {max_iter} iterations"
        )
    total_seconds = time.perf_counter() - started
    return _solution_payload(
        system,
        phi,
        k_eff=k_eff,
        iterations=iteration,
        inner_iterations=inner_counts,
        converged=True,
        timings={
            "factorization": factorization_seconds,
            "iteration": total_seconds - factorization_seconds,
            "total": total_seconds,
        },
    )


def solve_multigroup_2d(
    model: CylindricalLayeredModel2D,
    dr: float | None = 3.0,
    dz: float | None = 3.0,
    x_insert: float = 0.0,
    phi0=None,
    max_iter: int = 200,
    tol: float = 1e-6,
    *,
    mesh: CylindricalMesh2D | None = None,
    spacing: ConcentricMeshSpacing | None = None,
    source_tol: float = 1.0e-3,
    max_inner_iter: int = 100,
    inner_tol: float = 1.0e-4,
):
    if mesh is not None and spacing is not None:
        raise ValueError("Pass either mesh or spacing, not both")
    if mesh is None:
        if spacing is None:
            if dr is None or dz is None:
                spacing = ConcentricMeshSpacing()
            else:
                spacing = _legacy_spacing(dr, dz)
        mesh = build_concentric_mesh(model, spacing)
    system = build_multigroup_2d_system(
        model,
        mesh=mesh,
        x_insert=x_insert,
    )
    return solve_multigroup_system(
        system,
        phi0=phi0,
        max_iter=max_iter,
        tol=tol,
        source_tol=source_tol,
        max_inner_iter=max_inner_iter,
        inner_tol=inner_tol,
    )


def solve_multigroup_2d_global_reference(
    system: MultiGroupDiffusionSystem,
    *,
    phi0: Any = None,
    max_iter: int = 200,
    tol: float = 1.0e-8,
    source_tol: float = 1.0e-3,
) -> dict[str, Any]:
    """Solve the monolithic formulation for small verification problems."""
    started = time.perf_counter()
    loss, fission = _assemble_global_matrices(system)
    phi, _ = _normalize_fission(system, _initial_flux(system, phi0))
    solve_loss = spla.factorized(loss.tocsc())
    factorization_seconds = time.perf_counter() - started
    k_eff = 1.0
    converged = False

    for iteration in range(1, max_iter + 1):
        old_fission = _fission_density(system, phi)
        vector = phi.T.reshape(-1)
        solved = np.asarray(solve_loss((fission @ vector) / k_eff))
        phi_new = np.maximum(
            solved.reshape(system.group_count, system.cell_count).T,
            0.0,
        )
        phi_new, production = _normalize_fission(system, phi_new)
        k_new = k_eff * production
        source_error = float(
            np.dot(
                system.mesh.volumes,
                np.abs(_fission_density(system, phi_new) - old_fission),
            )
        )
        k_error = abs(k_new - k_eff) / max(abs(k_new), 1.0e-30)
        phi = phi_new
        k_eff = k_new
        if iteration > 1 and k_error < tol and source_error < source_tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"Global reference power iteration did not converge in {max_iter} iterations"
        )
    total_seconds = time.perf_counter() - started
    return _solution_payload(
        system,
        phi,
        k_eff=k_eff,
        iterations=iteration,
        inner_iterations=[],
        converged=True,
        timings={
            "factorization": factorization_seconds,
            "iteration": total_seconds - factorization_seconds,
            "total": total_seconds,
        },
    )


def scan_multigroup_rod_worth_2d(
    model: CylindricalLayeredModel2D,
    x_values: np.ndarray | list[float] | None = None,
    dr: float | None = 3.0,
    dz: float | None = 3.0,
    max_iter: int = 300,
    tol: float = 1e-6,
    warm_start: bool = True,
    *,
    mesh: CylindricalMesh2D | None = None,
    spacing: ConcentricMeshSpacing | None = None,
    clean_solution: dict[str, Any] | None = None,
    source_tol: float = 1.0e-3,
    max_inner_iter: int = 100,
    inner_tol: float = 1.0e-4,
):
    if x_values is None:
        x_arr = np.linspace(0.0, 1.0, 11)
    else:
        x_arr = np.asarray(x_values, dtype=float).reshape(-1)
    if x_arr.size == 0:
        raise ValueError("x_values must not be empty")
    if np.any((x_arr < 0.0) | (x_arr > 1.0)):
        raise ValueError("x_values must lie in [0, 1]")
    if mesh is not None and spacing is not None:
        raise ValueError("Pass either mesh or spacing, not both")
    if mesh is None:
        if spacing is None:
            spacing = (
                ConcentricMeshSpacing()
                if dr is None or dz is None
                else _legacy_spacing(dr, dz)
            )
        mesh = build_concentric_mesh(model, spacing)

    if clean_solution is None:
        clean_solution = solve_multigroup_2d(
            model,
            x_insert=0.0,
            mesh=mesh,
            max_iter=max_iter,
            tol=tol,
            source_tol=source_tol,
            max_inner_iter=max_inner_iter,
            inner_tol=inner_tol,
        )
    rho_clean_pcm = rho_pcm(clean_solution["k_eff"])
    phi_guess = clean_solution["phi_groups"] if warm_start else None

    k_values = np.zeros(x_arr.size, dtype=float)
    rho_total_pcm = np.zeros(x_arr.size, dtype=float)
    iterations: list[int] = []

    for index, x_insert in enumerate(x_arr):
        solution = solve_multigroup_2d(
            model,
            x_insert=float(x_insert),
            phi0=phi_guess,
            mesh=mesh,
            max_iter=max_iter,
            tol=tol,
            source_tol=source_tol,
            max_inner_iter=max_inner_iter,
            inner_tol=inner_tol,
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
            critical_insertion = float(
                x0 + (0.0 - rho0) * (x1 - x0) / (rho1 - rho0)
            )
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
