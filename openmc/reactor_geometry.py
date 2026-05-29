from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import openmc

from concentric_fuel import (
    ConcentricElementParameters,
    build_fuel_element_universe as build_concentric_fuel_element_universe,
    fuel_element_total_height_cm as concentric_fuel_element_total_height_cm,
    make_default_materials as make_concentric_default_materials,
)
from fuel_element import (
    build_fuel_element_universe as build_involute_fuel_element_universe,
    default_parameters as default_fuel_element_parameters,
    fuel_element_total_height_cm as involute_fuel_element_total_height_cm,
    make_default_materials as make_involute_default_materials,
)
from involutes import InvoluteElementParameters
from ploting import export_and_render_plots, reactor_plots


FuelElementParameters = InvoluteElementParameters | ConcentricElementParameters


def _fuel_element_total_height_cm(fuel_element: FuelElementParameters) -> float:
    if isinstance(fuel_element, ConcentricElementParameters):
        return concentric_fuel_element_total_height_cm(fuel_element)
    return involute_fuel_element_total_height_cm(fuel_element)


def _make_fuel_materials(
    fuel_element: FuelElementParameters,
) -> tuple[openmc.Materials, dict[str, openmc.Material]]:
    if isinstance(fuel_element, ConcentricElementParameters):
        return make_concentric_default_materials(fuel_element)
    return make_involute_default_materials(fuel_element)


def _build_fuel_universe(
    fuel_element: FuelElementParameters,
    material_map: dict[str, openmc.Material],
    rod_insertion: float,
    external_boundary_type: str | None,
) -> tuple[openmc.Universe, dict[str, Any]]:
    if isinstance(fuel_element, ConcentricElementParameters):
        return build_concentric_fuel_element_universe(
            fuel_element,
            material_map,
            rod_insertion=rod_insertion,
            external_boundary_type=external_boundary_type,
        )
    return build_involute_fuel_element_universe(
        fuel_element,
        material_map,
        rod_insertion=rod_insertion,
        external_boundary_type=external_boundary_type,
    )


@dataclass(frozen=True)
class ReactorTankParameters:
    d2o_tank_radius_cm: float = 65.0
    h2o_tank_radius_cm: float = 120.0
    h_d2o_tank_cm: float = 300.0
    h_h2o_tank_cm: float = 400.0

    def to_geometry_dict(self) -> dict[str, float]:
        values = asdict(self)
        values["h_total_cm"] = self.h_h2o_tank_cm
        return values


def validate_reactor_tanks(
    fuel_element: FuelElementParameters,
    tanks: ReactorTankParameters,
) -> None:
    if tanks.d2o_tank_radius_cm <= fuel_element.outer_radius_cm:
        raise ValueError("d2o_tank_radius_cm must be larger than the fuel element outer radius")
    if tanks.h2o_tank_radius_cm <= tanks.d2o_tank_radius_cm:
        raise ValueError("h2o_tank_radius_cm must be larger than d2o_tank_radius_cm")
    if tanks.h_d2o_tank_cm < _fuel_element_total_height_cm(fuel_element):
        raise ValueError("h_d2o_tank_cm must be at least as tall as the fuel element stack")
    if tanks.h_h2o_tank_cm < tanks.h_d2o_tank_cm:
        raise ValueError("h_h2o_tank_cm must be at least as tall as h_d2o_tank_cm")


def make_reactor_materials(
    fuel_element: FuelElementParameters,
) -> tuple[openmc.Materials, dict[str, openmc.Material]]:
    materials, material_map = _make_fuel_materials(fuel_element)

    light_water = openmc.Material(name="Light water tank")
    light_water.set_density("g/cm3", 0.997)
    light_water.add_nuclide("H1", 2.0)
    light_water.add_nuclide("O16", 1.0)
    light_water.add_s_alpha_beta("c_H_in_H2O")
    light_water.temperature = 300.0

    materials.append(light_water)
    material_map["light_water"] = light_water
    return materials, material_map


def build_reactor_universe(
    fuel_element: FuelElementParameters,
    tanks: ReactorTankParameters,
    rod_insertion: float = 0.0,
    material_map: dict[str, openmc.Material] | None = None,
) -> tuple[openmc.Universe, dict[str, Any]]:
    validate_reactor_tanks(fuel_element, tanks)
    if material_map is None:
        _, material_map = make_reactor_materials(fuel_element)

    fuel_universe, fuel_metadata = _build_fuel_universe(
        fuel_element,
        material_map,
        rod_insertion=rod_insertion,
        external_boundary_type=None,
    )

    fuel_total_half = 0.5 * _fuel_element_total_height_cm(fuel_element)
    d2o_half = 0.5 * tanks.h_d2o_tank_cm
    total_half = 0.5 * tanks.h_h2o_tank_cm

    fuel_outer_cyl = openmc.ZCylinder(r=fuel_element.outer_radius_cm)
    d2o_tank_cyl = openmc.ZCylinder(r=tanks.d2o_tank_radius_cm)
    h2o_tank_cyl = openmc.ZCylinder(r=tanks.h2o_tank_radius_cm, boundary_type="vacuum")

    z_fuel_bottom = openmc.ZPlane(z0=-fuel_total_half)
    z_fuel_top = openmc.ZPlane(z0=fuel_total_half)
    z_d2o_bottom = openmc.ZPlane(z0=-d2o_half)
    z_d2o_top = openmc.ZPlane(z0=d2o_half)
    z_total_bottom = openmc.ZPlane(z0=-total_half, boundary_type="vacuum")
    z_total_top = openmc.ZPlane(z0=total_half, boundary_type="vacuum")

    fuel_element_region = -fuel_outer_cyl & +z_fuel_bottom & -z_fuel_top
    d2o_region = -d2o_tank_cyl & +z_d2o_bottom & -z_d2o_top & ~fuel_element_region
    h2o_region = (
        -h2o_tank_cyl
        & +z_total_bottom
        & -z_total_top
        & ~(-d2o_tank_cyl & +z_d2o_bottom & -z_d2o_top)
    )

    cells = [
        openmc.Cell(name="fuel_element", fill=fuel_universe, region=fuel_element_region),
        openmc.Cell(name="d2o_tank", fill=material_map["moderator"], region=d2o_region),
        openmc.Cell(name="h2o_tank", fill=material_map["light_water"], region=h2o_region),
    ]

    universe = openmc.Universe(name="Reactor geometry", cells=cells)
    metadata = {
        "fuel_element": fuel_metadata,
        "tank_report": tanks.to_geometry_dict(),
        "total_height_cm": tanks.h_h2o_tank_cm,
    }
    return universe, metadata


def build_reactor_model(
    fuel_element: FuelElementParameters,
    tanks: ReactorTankParameters,
    rod_insertion: float = 0.0,
) -> tuple[openmc.Model, dict[str, Any]]:
    materials, material_map = make_reactor_materials(fuel_element)
    universe, metadata = build_reactor_universe(
        fuel_element,
        tanks,
        rod_insertion=rod_insertion,
        material_map=material_map,
    )
    model = openmc.Model(geometry=openmc.Geometry(universe), materials=materials)
    metadata["materials"] = materials
    return model, metadata


def export_reactor_artifacts(
    model: openmc.Model,
    fuel_element: FuelElementParameters,
    tanks: ReactorTankParameters,
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

    plots = reactor_plots(
        geometry,
        fuel_element_outer_radius_cm=fuel_element.outer_radius_cm,
        d2o_tank_radius_cm=tanks.d2o_tank_radius_cm,
        h2o_tank_radius_cm=tanks.h2o_tank_radius_cm,
        d2o_tank_height_cm=tanks.h_d2o_tank_cm,
        total_height_cm=tanks.h_h2o_tank_cm,
    )
    export_and_render_plots(model, plots, output_dir, openmc_exec=openmc_exec)


def default_tank_parameters() -> ReactorTankParameters:
    return ReactorTankParameters(
        d2o_tank_radius_cm=65.0,
        h2o_tank_radius_cm=120.0,
        h_d2o_tank_cm=300.0,
        h_h2o_tank_cm=400.0,
    )


if __name__ == "__main__":
    fuel_element = default_fuel_element_parameters()
    tanks = default_tank_parameters()
    model, metadata = build_reactor_model(fuel_element, tanks)
    print(metadata["tank_report"])

    output_dir = Path.cwd() / "build" / "reactor"
    export_reactor_artifacts(model, fuel_element, tanks, output_dir)
    print(f"Reactor geometry XML and plots XML written to {output_dir}")