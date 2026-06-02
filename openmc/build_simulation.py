from __future__ import annotations

import numpy as np
import openmc

from reactor_geometry import build_reactor_model


def estimate_particles_per_mesh_bin(
    particles: int,
    batches: int,
    inactive: int,
    mesh_shape: tuple[int, int],
) -> float:
    active_batches = max(0, batches - inactive)
    mesh_bins = mesh_shape[0] * mesh_shape[1]
    if mesh_bins <= 0 or active_batches <= 0:
        return 0.0
    return particles * active_batches / mesh_bins


def build_entropy_mesh(
    outer_radius_cm: float,
    active_height_cm: float,
    particles: int,
    target_source_sites_per_cell: float = 20.0,
) -> openmc.RegularMesh:
    radial_span = 2.0 * outer_radius_cm
    axial_span = active_height_cm
    bounding_volume = radial_span * radial_span * axial_span
    target_cells = max(1.0, particles / target_source_sites_per_cell)
    cells_per_cm = (target_cells / bounding_volume) ** (1.0 / 3.0)

    nx = max(4, int(round(radial_span * cells_per_cm)))
    ny = max(4, int(round(radial_span * cells_per_cm)))
    nz = max(8, int(round(axial_span * cells_per_cm)))

    entropy_mesh = openmc.RegularMesh()
    entropy_mesh.lower_left = (
        -outer_radius_cm,
        -outer_radius_cm,
        -0.5 * active_height_cm,
    )
    entropy_mesh.upper_right = (
        outer_radius_cm,
        outer_radius_cm,
        0.5 * active_height_cm,
    )
    entropy_mesh.dimension = (nx, ny, nz)
    return entropy_mesh


def _build_axial_mesh(
    mesh_shape: tuple[int, int],
    half_width_cm: float,
    half_height_cm: float,
) -> openmc.RegularMesh:
    mesh = openmc.RegularMesh()
    mesh.dimension = (mesh_shape[0], 1, mesh_shape[1])
    mesh.lower_left = (-half_width_cm, -1.0, -half_height_cm)
    mesh.upper_right = (half_width_cm, 1.0, half_height_cm)
    return mesh


def _build_mesh_tally(name: str, mesh: openmc.RegularMesh) -> openmc.Tally:
    tally = openmc.Tally(name=name)
    tally.filters = [openmc.MeshFilter(mesh)]
    tally.scores = ["flux", "fission", "nu-fission"]
    return tally


def build_eigenvalue_model(
    fuel_parameters,
    reactor_tank_parameters,
    *,
    rod_insertion: float = 0.0,
    particles: int = 12000,
    batches: int = 100,
    inactive: int = 20,
    global_mesh_shape: tuple[int, int] | None = None,
    fuel_mesh_shape: tuple[int, int] | None = None,
) -> tuple[openmc.Model, dict]:
    rod_insertion = float(np.clip(rod_insertion, 0.0, 1.0))
    model, metadata = build_reactor_model(
        fuel_parameters,
        reactor_tank_parameters,
        rod_insertion=rod_insertion,
    )

    total_half = 0.5 * reactor_tank_parameters.h_h2o_tank_cm
    active_half = 0.5 * fuel_parameters.h_active_cm

    source = openmc.IndependentSource(
        space=openmc.stats.CylindricalIndependent(
            r=openmc.stats.PowerLaw(
                fuel_parameters.inner_radius_cm,
                fuel_parameters.outer_radius_cm,
                1.0,
            ),
            phi=openmc.stats.Uniform(0.0, 2.0 * np.pi),
            z=openmc.stats.Uniform(-active_half, active_half),
        ),
        constraints={"fissionable": True},
    )

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.source = source
    settings.source_rejection_fraction = 0.01
    settings.temperature = {"method": "interpolation"}

    entropy_mesh = build_entropy_mesh(
        fuel_parameters.outer_radius_cm,
        fuel_parameters.h_active_cm,
        particles,
    )
    settings.entropy_mesh = entropy_mesh
    model.settings = settings

    tallies: list[openmc.Tally] = []

    if global_mesh_shape is not None:
        global_mesh = _build_axial_mesh(
            global_mesh_shape,
            reactor_tank_parameters.h2o_tank_radius_cm,
            total_half,
        )
        tallies.append(_build_mesh_tally("global-mesh", global_mesh))
        metadata["global_mesh_shape"] = tuple(global_mesh.dimension)
        metadata["global_mesh_histories_per_bin"] = estimate_particles_per_mesh_bin(
            particles,
            batches,
            inactive,
            global_mesh_shape,
        )

    if fuel_mesh_shape is not None:
        fuel_mesh = _build_axial_mesh(
            fuel_mesh_shape,
            fuel_parameters.outer_radius_cm,
            active_half,
        )
        tallies.append(_build_mesh_tally("fuel-mesh", fuel_mesh))
        metadata["fuel_mesh_shape"] = tuple(fuel_mesh.dimension)
        metadata["fuel_mesh_histories_per_bin"] = estimate_particles_per_mesh_bin(
            particles,
            batches,
            inactive,
            fuel_mesh_shape,
        )

    if tallies:
        model.tallies = openmc.Tallies(tallies)

    metadata["entropy_mesh_shape"] = tuple(entropy_mesh.dimension)
    metadata["tally_names"] = [tally.name for tally in tallies]
    metadata["particles"] = particles
    metadata["batches"] = batches
    metadata["inactive"] = inactive
    return model, metadata