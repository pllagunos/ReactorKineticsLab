from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import openmc


@dataclass(frozen=True)
class InvoluteElementParameters:
    plate_count: int = 113
    plate_thickness_cm: float = 0.15
    coolant_gap_cm: float = 0.20
    inner_radius_cm: float = 20.0
    outer_radius_cm: float = 50.0
    control_rod_radius_cm: float = 4.0
    h_active_cm: float = 200.0
    lower_plenum_cm: float = 50.0
    upper_plenum_cm: float = 50.0
    segments_per_plate: int = 14
    base_radius_cm: float | None = None

    def resolved_base_radius_cm(self) -> float:
        if self.base_radius_cm is not None:
            return self.base_radius_cm
        pitch = self.plate_thickness_cm + self.coolant_gap_cm
        return self.plate_count * pitch / (2.0 * np.pi)

    def to_geometry_dict(self) -> dict[str, float | int]:
        values = asdict(self)
        values["base_radius_cm"] = self.resolved_base_radius_cm()
        values["h_total_cm"] = self.h_active_cm + self.lower_plenum_cm + self.upper_plenum_cm
        return values


def validate_parameters(parameters: InvoluteElementParameters) -> None:
    base_radius_cm = parameters.resolved_base_radius_cm()
    if parameters.plate_count < 1:
        raise ValueError("plate_count must be positive")
    if parameters.segments_per_plate < 2:
        raise ValueError("segments_per_plate must be at least 2")
    if parameters.plate_thickness_cm <= 0.0:
        raise ValueError("plate_thickness_cm must be positive")
    if parameters.coolant_gap_cm <= 0.0:
        raise ValueError("coolant_gap_cm must be positive")
    if base_radius_cm <= 0.0:
        raise ValueError("base_radius_cm must be positive")
    if base_radius_cm >= parameters.inner_radius_cm:
        raise ValueError("base_radius_cm must remain inside the central moderator hole")
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


def involute_theta_bounds(parameters: InvoluteElementParameters) -> tuple[float, float]:
    base_radius_cm = parameters.resolved_base_radius_cm()
    theta_inner = np.sqrt((parameters.inner_radius_cm / base_radius_cm) ** 2 - 1.0)
    theta_outer = np.sqrt((parameters.outer_radius_cm / base_radius_cm) ** 2 - 1.0)
    return theta_inner, theta_outer


def involute_point(base_radius_cm: float, theta: float) -> np.ndarray:
    return base_radius_cm * np.array(
        [np.cos(theta) + theta * np.sin(theta), np.sin(theta) - theta * np.cos(theta)]
    )


def involute_tangent(base_radius_cm: float, theta: float) -> np.ndarray:
    return base_radius_cm * theta * np.array([np.cos(theta), np.sin(theta)])


def rotated_points(points: np.ndarray, angle_rad: float) -> np.ndarray:
    rotation = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]]
    )
    return points @ rotation.T


def involute_strip_points(
    parameters: InvoluteElementParameters, plate_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_parameters(parameters)
    base_radius_cm = parameters.resolved_base_radius_cm()
    theta_inner, theta_outer = involute_theta_bounds(parameters)
    theta_values = np.linspace(theta_inner, theta_outer, parameters.segments_per_plate + 1)

    centerline = np.stack([involute_point(base_radius_cm, theta) for theta in theta_values])
    tangents = np.stack([involute_tangent(base_radius_cm, theta) for theta in theta_values])
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / tangent_norms

    half_thickness = 0.5 * parameters.plate_thickness_cm
    inner_boundary = centerline - half_thickness * normals
    outer_boundary = centerline + half_thickness * normals

    angle_step = 2.0 * np.pi / parameters.plate_count
    rotation_angle = plate_index * angle_step
    return (
        rotated_points(centerline, rotation_angle),
        rotated_points(inner_boundary, rotation_angle),
        rotated_points(outer_boundary, rotation_angle),
    )


def plate_polygon_points(
    parameters: InvoluteElementParameters, plate_index: int
) -> list[tuple[float, float]]:
    _, inner_boundary, outer_boundary = involute_strip_points(parameters, plate_index)
    polygon_points = np.vstack([inner_boundary, outer_boundary[::-1]])
    return [tuple(point) for point in polygon_points]


def plate_region(parameters: InvoluteElementParameters, plate_index: int) -> openmc.Region:
    polygon = openmc.model.Polygon(plate_polygon_points(parameters, plate_index), basis="xy")
    return polygon.region


def all_plate_regions(parameters: InvoluteElementParameters) -> list[openmc.Region]:
    return [plate_region(parameters, plate_index) for plate_index in range(parameters.plate_count)]


def plate_cells(
    parameters: InvoluteElementParameters,
    fuel_material: openmc.Material,
    z_bottom: openmc.ZPlane,
    z_top: openmc.ZPlane,
) -> tuple[list[openmc.Cell], openmc.Region]:
    cells: list[openmc.Cell] = []
    active_union: openmc.Region | None = None
    for plate_index, region in enumerate(all_plate_regions(parameters)):
        extruded_region = region & +z_bottom & -z_top
        cells.append(
            openmc.Cell(
                name=f"fuel_plate_{plate_index + 1}",
                fill=fuel_material,
                region=extruded_region,
            )
        )
        active_union = extruded_region if active_union is None else active_union | extruded_region

    if active_union is None:
        raise ValueError("No fuel plate regions were generated")
    return cells, active_union


def spiral_summary(parameters: InvoluteElementParameters) -> dict[str, float | int]:
    validate_parameters(parameters)
    base_radius_cm = parameters.resolved_base_radius_cm()
    theta_inner, theta_outer = involute_theta_bounds(parameters)
    arc_length = 0.5 * base_radius_cm * (
        theta_outer * np.sqrt(1.0 + theta_outer**2)
        + np.arcsinh(theta_outer)
        - theta_inner * np.sqrt(1.0 + theta_inner**2)
        - np.arcsinh(theta_inner)
    )
    uranium_area_cm2 = parameters.plate_count * arc_length * parameters.plate_thickness_cm
    annulus_area_cm2 = np.pi * (parameters.outer_radius_cm**2 - parameters.inner_radius_cm**2)
    return {
        "plate_count": parameters.plate_count,
        "base_radius_cm": base_radius_cm,
        "theta_inner": theta_inner,
        "theta_outer": theta_outer,
        "single_plate_length_cm": arc_length,
        "uranium_area_fraction": uranium_area_cm2 / annulus_area_cm2,
    }


def build_parameter_report(parameters: InvoluteElementParameters) -> dict[str, float | int]:
    report = parameters.to_geometry_dict()
    report.update(spiral_summary(parameters))
    return report


__all__ = [
    "InvoluteElementParameters",
    "all_plate_regions",
    "build_parameter_report",
    "involute_strip_points",
    "plate_polygon_points",
    "plate_cells",
    "plate_region",
    "spiral_summary",
    "validate_parameters",
]