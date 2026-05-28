from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import openmc

from involutes import (
    InvoluteElementParameters,
    build_parameter_report,
    involute_strip_points,
    plate_cells,
    spiral_summary,
    validate_parameters,
)
from ploting import fuel_element_plots
from ploting import export_and_render_plots


DEFAULT_FUEL_DENSITY_G_PER_CM3 = 19.05


def _z_plane(z0: float, boundary_type: str | None = None) -> openmc.ZPlane:
    kwargs = {"boundary_type": boundary_type} if boundary_type is not None else {}
    return openmc.ZPlane(z0=z0, **kwargs)


def _z_cylinder(radius_cm: float, boundary_type: str | None = None) -> openmc.ZCylinder:
    kwargs = {"boundary_type": boundary_type} if boundary_type is not None else {}
    return openmc.ZCylinder(r=radius_cm, **kwargs)


def fuel_element_total_height_cm(parameters: InvoluteElementParameters) -> float:
    return parameters.lower_plenum_cm + parameters.h_active_cm + parameters.upper_plenum_cm


def fuel_volume_cm3(parameters: InvoluteElementParameters) -> float:
    summary = spiral_summary(parameters)
    single_plate_length_cm = float(summary["single_plate_length_cm"])
    return (
        parameters.plate_count
        * parameters.plate_thickness_cm
        * single_plate_length_cm
        * parameters.h_active_cm
    )


def fuel_mass_kg(
    parameters: InvoluteElementParameters,
    density_g_per_cm3: float = DEFAULT_FUEL_DENSITY_G_PER_CM3,
) -> float:
    if density_g_per_cm3 <= 0.0:
        raise ValueError("density_g_per_cm3 must be positive")
    return fuel_volume_cm3(parameters) * density_g_per_cm3 / 1000.0


def make_default_materials() -> tuple[openmc.Materials, dict[str, openmc.Material]]:
    fuel = openmc.Material(name="Natural uranium metal")
    fuel.set_density("g/cm3", DEFAULT_FUEL_DENSITY_G_PER_CM3)
    fuel.add_element("U", 1.0, enrichment=20.0)
    fuel.temperature = 600.0

    heavy_water = openmc.Material(name="Heavy water moderator")
    heavy_water.set_density("g/cm3", 1.105)
    heavy_water.add_nuclide("H2", 2.0)
    heavy_water.add_nuclide("O16", 1.0)
    heavy_water.add_s_alpha_beta("c_D_in_D2O")
    heavy_water.temperature = 300.0

    control_rod = openmc.Material(name="B4C control rod")
    control_rod.set_density("g/cm3", 2.52)
    control_rod.add_nuclide("B10", 3.8)
    control_rod.add_nuclide("B11", 0.2)
    control_rod.add_element("C", 1.0)
    control_rod.temperature = 300.0

    materials = openmc.Materials([fuel, heavy_water, control_rod])
    return materials, {
        "fuel": fuel,
        "moderator": heavy_water,
        "control_rod": control_rod,
    }


def build_fuel_element_universe(
    parameters: InvoluteElementParameters,
    material_map: dict[str, openmc.Material],
    rod_insertion: float = 0.0,
    external_boundary_type: str | None = "vacuum",
) -> tuple[openmc.Universe, dict[str, Any]]:
    validate_parameters(parameters)
    rod_insertion = float(np.clip(rod_insertion, 0.0, 1.0))

    total_height_cm = fuel_element_total_height_cm(parameters)
    total_half = 0.5 * total_height_cm
    active_half = 0.5 * parameters.h_active_cm
    z_bottom = _z_plane(z0=-total_half, boundary_type=external_boundary_type)
    z_top = _z_plane(z0=total_half, boundary_type=external_boundary_type)
    z_active_bottom = _z_plane(z0=-active_half)
    z_active_top = _z_plane(z0=active_half)
    z_rod_tip = _z_plane(z0=active_half - rod_insertion * parameters.h_active_cm)

    outer_cyl = _z_cylinder(parameters.outer_radius_cm, boundary_type=external_boundary_type)
    rod_cyl = _z_cylinder(parameters.control_rod_radius_cm)

    fuel_cells, fuel_region = plate_cells(
        parameters,
        material_map["fuel"],
        z_active_bottom,
        z_active_top,
    )

    rod_region = -rod_cyl & +z_rod_tip & -z_top
    central_moderator_region = -rod_cyl & +z_bottom & -z_rod_tip
    bulk_moderator_region = -outer_cyl & +z_bottom & -z_top & ~fuel_region & ~rod_region & ~central_moderator_region

    cells = [
        openmc.Cell(name="control_rod", fill=material_map["control_rod"], region=rod_region),
        openmc.Cell(
            name="central_moderator_channel",
            fill=material_map["moderator"],
            region=central_moderator_region,
        ),
        openmc.Cell(
            name="heavy_water_coolant_and_moderator",
            fill=material_map["moderator"],
            region=bulk_moderator_region,
        ),
        *fuel_cells,
    ]

    universe = openmc.Universe(name="Involute fuel element", cells=cells)
    metadata = {
        "total_height_cm": total_height_cm,
        "rod_tip_z_cm": active_half - rod_insertion * parameters.h_active_cm,
        "fuel_cell_count": len(fuel_cells),
        "report": build_parameter_report(parameters),
        "spiral_summary": spiral_summary(parameters),
    }
    return universe, metadata


def build_fuel_element_geometry(
    parameters: InvoluteElementParameters,
    material_map: dict[str, openmc.Material],
    rod_insertion: float = 0.0,
) -> tuple[openmc.Geometry, dict[str, Any]]:
    universe, metadata = build_fuel_element_universe(
        parameters,
        material_map,
        rod_insertion=rod_insertion,
    )
    return openmc.Geometry(universe), metadata


def build_fuel_element_model(
    parameters: InvoluteElementParameters,
    rod_insertion: float = 0.0,
) -> tuple[openmc.Model, dict[str, Any]]:
    materials, material_map = make_default_materials()
    geometry, metadata = build_fuel_element_geometry(
        parameters,
        material_map,
        rod_insertion=rod_insertion,
    )
    model = openmc.Model(geometry=geometry, materials=materials)
    metadata["materials"] = materials
    return model, metadata


def quick_layout_plot(
    parameters: InvoluteElementParameters,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    for plate_index in range(parameters.plate_count):
        _, inner_boundary, outer_boundary = involute_strip_points(parameters, plate_index)
        polygon = np.vstack([inner_boundary, outer_boundary[::-1]])
        ax.fill(polygon[:, 0], polygon[:, 1], color="tab:red", alpha=0.45, linewidth=0.0)

    for radius_cm, color, linestyle, label in (
        (parameters.control_rod_radius_cm, "black", "--", "control rod"),
        (parameters.inner_radius_cm, "tab:blue", "-", "fuel inner radius"),
        (parameters.outer_radius_cm, "tab:green", "-", "fuel outer radius"),
    ):
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                radius_cm,
                fill=False,
                linestyle=linestyle,
                color=color,
                linewidth=1.2,
                label=label,
            )
        )

    ax.set_aspect("equal")
    ax.set_xlim(-1.1 * parameters.outer_radius_cm, 1.1 * parameters.outer_radius_cm)
    ax.set_ylim(-1.1 * parameters.outer_radius_cm, 1.1 * parameters.outer_radius_cm)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title("Fast involute fuel-element layout")
    ax.legend(loc="upper right")
    return ax


def export_geometry_artifacts(
    model: openmc.Model,
    parameters: InvoluteElementParameters,
    output_dir: Path,
    openmc_exec: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    materials = model.materials
    geometry = model.geometry
    if materials is None or geometry is None:
        raise ValueError("Model must contain materials and geometry")

    materials.export_to_xml(path=output_dir / "materials.xml")
    geometry.export_to_xml(path=output_dir / "geometry.xml")

    plots = fuel_element_plots(
        geometry,
        outer_radius_cm=parameters.outer_radius_cm,
        total_height_cm=parameters.lower_plenum_cm + parameters.h_active_cm + parameters.upper_plenum_cm,
    )
    export_and_render_plots(model, plots, output_dir, openmc_exec=openmc_exec)


def default_parameters() -> InvoluteElementParameters:
    # return InvoluteElementParameters(
    #     plate_count=113,
    #     plate_thickness_cm=0.15,
    #     coolant_gap_cm=0.20,
    #     inner_radius_cm=20.0,
    #     outer_radius_cm=50.0,
    #     control_rod_radius_cm=4.0,
    #     h_active_cm=200.0,
    #     lower_plenum_cm=50.0,
    #     upper_plenum_cm=50.0,
    #     segments_per_plate=100,
    #     base_radius_cm=18.0,
    # )
    return InvoluteElementParameters(
        plate_count=25,
        plate_thickness_cm=2.0,
        coolant_gap_cm=2.0,
        inner_radius_cm=20.0,
        outer_radius_cm=50.0,
        control_rod_radius_cm=4.0,
        h_active_cm=200.0,
        lower_plenum_cm=50.0,
        upper_plenum_cm=50.0,
        segments_per_plate=24,
        base_radius_cm=18.0,
    )


if __name__ == "__main__":
    parameters = default_parameters()
    validate_parameters(parameters)
    model, metadata = build_fuel_element_model(parameters)
    print(metadata["report"])
    print(f"Fuel mass: {fuel_mass_kg(parameters):.3f} kg")

    output_dir = Path.cwd() / "build" / "fuel_element"

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    quick_layout_plot(parameters, ax=ax)
    figure_path = output_dir / "fuel_element_layout.png"
    fig.savefig(figure_path, dpi=200)
    print(f"Geometry XML and plots XML written to {output_dir}")
    print(f"Fast layout preview written to {figure_path}")

    export_geometry_artifacts(model, parameters, output_dir)