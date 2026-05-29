from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import openmc

from fuel_element import make_default_materials as make_shared_materials
from ploting import export_and_render_plots, fuel_element_plots


def _z_plane(z0: float, boundary_type: str | None = None) -> openmc.ZPlane:
    kwargs = {"boundary_type": boundary_type} if boundary_type is not None else {}
    return openmc.ZPlane(z0=z0, **kwargs)


def _z_cylinder(radius_cm: float, boundary_type: str | None = None) -> openmc.ZCylinder:
    kwargs = {"boundary_type": boundary_type} if boundary_type is not None else {}
    return openmc.ZCylinder(r=radius_cm, **kwargs)


@dataclass(frozen=True)
class ConcentricElementParameters:
    ring_count: int = 6
    ring_thickness_cm: float = 1.5
    coolant_gap_cm: float = 4.5
    inner_radius_cm: float = 20.0
    outer_radius_cm: float = 60.0
    control_rod_radius_cm: float = 4.0
    h_active_cm: float = 350.0
    lower_plenum_cm: float = 50.0
    upper_plenum_cm: float = 50.0
    fuel_density_g_per_cm3: float = 12.2
    fuel_enrichment_wt_pct: float = 0.7

    def radial_span_cm(self) -> float:
        return self.ring_count * self.ring_thickness_cm + max(0, self.ring_count - 1) * self.coolant_gap_cm

    def edge_gap_cm(self) -> float:
        return 0.5 * ((self.outer_radius_cm - self.inner_radius_cm) - self.radial_span_cm())

    def to_geometry_dict(self) -> dict[str, float | int]:
        values = asdict(self)
        values["radial_span_cm"] = self.radial_span_cm()
        values["edge_gap_cm"] = self.edge_gap_cm()
        values["h_total_cm"] = fuel_element_total_height_cm(self)
        return values


def validate_parameters(parameters: ConcentricElementParameters) -> None:
    if parameters.ring_count < 1:
        raise ValueError("ring_count must be positive")
    if parameters.ring_thickness_cm <= 0.0:
        raise ValueError("ring_thickness_cm must be positive")
    if parameters.coolant_gap_cm <= 0.0:
        raise ValueError("coolant_gap_cm must be positive")
    if parameters.inner_radius_cm >= parameters.outer_radius_cm:
        raise ValueError("inner_radius_cm must be smaller than outer_radius_cm")
    if parameters.control_rod_radius_cm <= 0.0:
        raise ValueError("control_rod_radius_cm must be positive")
    if parameters.control_rod_radius_cm >= parameters.inner_radius_cm:
        raise ValueError("control_rod_radius_cm must be smaller than inner_radius_cm")
    if parameters.h_active_cm <= 0.0:
        raise ValueError("h_active_cm must be positive")
    if parameters.lower_plenum_cm < 0.0 or parameters.upper_plenum_cm < 0.0:
        raise ValueError("plenum heights cannot be negative")
    if parameters.fuel_density_g_per_cm3 <= 0.0:
        raise ValueError("fuel_density_g_per_cm3 must be positive")
    if not 0.0 < parameters.fuel_enrichment_wt_pct < 100.0:
        raise ValueError("fuel_enrichment_wt_pct must be between 0 and 100")
    if parameters.edge_gap_cm() < 0.0:
        raise ValueError(
            "Concentric annuli do not fit inside the requested radial envelope. "
            f"ring_count={parameters.ring_count}, ring_thickness_cm={parameters.ring_thickness_cm:.6g} cm, "
            f"coolant_gap_cm={parameters.coolant_gap_cm:.6g} cm require a radial span of "
            f"{parameters.radial_span_cm():.6g} cm, but only "
            f"{parameters.outer_radius_cm - parameters.inner_radius_cm:.6g} cm are available."
        )


def fuel_element_total_height_cm(parameters: ConcentricElementParameters) -> float:
    return parameters.lower_plenum_cm + parameters.h_active_cm + parameters.upper_plenum_cm


def ring_radii(parameters: ConcentricElementParameters) -> list[tuple[float, float]]:
    validate_parameters(parameters)
    radii: list[tuple[float, float]] = []
    ring_start_cm = parameters.inner_radius_cm + parameters.edge_gap_cm()
    pitch_cm = parameters.ring_thickness_cm + parameters.coolant_gap_cm
    for ring_index in range(parameters.ring_count):
        inner_radius_cm = ring_start_cm + ring_index * pitch_cm
        outer_radius_cm = inner_radius_cm + parameters.ring_thickness_cm
        radii.append((inner_radius_cm, outer_radius_cm))
    return radii


def fuel_area_cm2(parameters: ConcentricElementParameters) -> float:
    return float(
        np.pi * sum(outer_radius_cm**2 - inner_radius_cm**2 for inner_radius_cm, outer_radius_cm in ring_radii(parameters))
    )


def fuel_volume_cm3(parameters: ConcentricElementParameters) -> float:
    return fuel_area_cm2(parameters) * parameters.h_active_cm


def fuel_mass_kg(parameters: ConcentricElementParameters) -> float:
    return fuel_volume_cm3(parameters) * parameters.fuel_density_g_per_cm3 / 1000.0


def ring_summary(parameters: ConcentricElementParameters) -> dict[str, float | int]:
    validate_parameters(parameters)
    annulus_area_cm2 = np.pi * (parameters.outer_radius_cm**2 - parameters.inner_radius_cm**2)
    radii = ring_radii(parameters)
    return {
        "ring_count": parameters.ring_count,
        "ring_thickness_cm": parameters.ring_thickness_cm,
        "coolant_gap_cm": parameters.coolant_gap_cm,
        "radial_span_cm": parameters.radial_span_cm(),
        "edge_gap_cm": parameters.edge_gap_cm(),
        "first_ring_inner_radius_cm": radii[0][0],
        "last_ring_outer_radius_cm": radii[-1][1],
        "uranium_area_fraction": fuel_area_cm2(parameters) / annulus_area_cm2,
        "uranium_volume_cm3": fuel_volume_cm3(parameters),
        "uranium_mass_kg": fuel_mass_kg(parameters),
    }


def build_parameter_report(parameters: ConcentricElementParameters) -> dict[str, float | int]:
    report = parameters.to_geometry_dict()
    report.update(ring_summary(parameters))
    return report


def make_default_materials(
    parameters: ConcentricElementParameters,
) -> tuple[openmc.Materials, dict[str, openmc.Material]]:
    return make_shared_materials(parameters)


def ring_cells(
    parameters: ConcentricElementParameters,
    fuel_material: openmc.Material,
    z_bottom: openmc.ZPlane,
    z_top: openmc.ZPlane,
) -> tuple[list[openmc.Cell], openmc.Region]:
    cells: list[openmc.Cell] = []
    active_union: openmc.Region | None = None
    for ring_index, (inner_radius_cm, outer_radius_cm) in enumerate(ring_radii(parameters)):
        inner_cyl = _z_cylinder(inner_radius_cm)
        outer_cyl = _z_cylinder(outer_radius_cm)
        extruded_region = +inner_cyl & -outer_cyl & +z_bottom & -z_top
        cells.append(
            openmc.Cell(
                name=f"fuel_ring_{ring_index + 1}",
                fill=fuel_material,
                region=extruded_region,
            )
        )
        active_union = extruded_region if active_union is None else active_union | extruded_region

    if active_union is None:
        raise ValueError("No concentric fuel regions were generated")
    return cells, active_union


def build_fuel_element_universe(
    parameters: ConcentricElementParameters,
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

    fuel_cells, fuel_region = ring_cells(
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

    universe = openmc.Universe(name="Concentric annular fuel element", cells=cells)
    metadata = {
        "total_height_cm": total_height_cm,
        "rod_tip_z_cm": active_half - rod_insertion * parameters.h_active_cm,
        "fuel_cell_count": len(fuel_cells),
        "report": build_parameter_report(parameters),
        "ring_summary": ring_summary(parameters),
    }
    return universe, metadata


def build_fuel_element_model(
    parameters: ConcentricElementParameters,
    rod_insertion: float = 0.0,
) -> tuple[openmc.Model, dict[str, Any]]:
    materials, material_map = make_default_materials(parameters)
    universe, metadata = build_fuel_element_universe(
        parameters,
        material_map,
        rod_insertion=rod_insertion,
    )
    model = openmc.Model(geometry=openmc.Geometry(universe), materials=materials)
    metadata["materials"] = materials
    return model, metadata


def quick_layout_plot(
    parameters: ConcentricElementParameters,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    for inner_radius_cm, outer_radius_cm in ring_radii(parameters):
        ax.add_patch(
            Wedge(
                center=(0.0, 0.0),
                r=outer_radius_cm,
                theta1=0.0,
                theta2=360.0,
                width=outer_radius_cm - inner_radius_cm,
                facecolor="tab:red",
                edgecolor="none",
                alpha=0.45,
            )
        )

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
    ax.set_title("Fast concentric annular fuel-element layout")
    ax.legend(loc="upper right")
    return ax


def export_geometry_artifacts(
    model: openmc.Model,
    parameters: ConcentricElementParameters,
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
        total_height_cm=fuel_element_total_height_cm(parameters),
    )
    export_and_render_plots(model, plots, output_dir, openmc_exec=openmc_exec)


def default_parameters() -> ConcentricElementParameters:
    return ConcentricElementParameters()


__all__ = [
    "ConcentricElementParameters",
    "build_fuel_element_model",
    "build_fuel_element_universe",
    "build_parameter_report",
    "default_parameters",
    "export_geometry_artifacts",
    "fuel_element_total_height_cm",
    "fuel_mass_kg",
    "fuel_volume_cm3",
    "make_default_materials",
    "quick_layout_plot",
    "ring_radii",
    "ring_summary",
    "validate_parameters",
]