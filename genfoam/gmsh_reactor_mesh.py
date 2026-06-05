from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CASE_DIR / "constant" / "generated"
DEFAULT_MESH_FILE = DEFAULT_OUTPUT_DIR / "concentric_reactor_wedge.msh"
DEFAULT_MANIFEST_FILE = DEFAULT_OUTPUT_DIR / "concentric_reactor_mesh_manifest.json"
WEDGE_MESH_KIND = "axisymmetric-wedge"
FULL_3D_EXPERIMENTAL_MESH_KIND = "full-3d-experimental"
_BOUNDARY_BLOCK_PATTERN = re.compile(
    r"(^\s*(?P<name>[A-Za-z0-9_]+)\s*\n\s*\{)(?P<body>.*?)(^\s*\})",
    re.MULTILINE | re.DOTALL,
)


def _import_gmsh() -> Any:
    try:
        import gmsh  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The gmsh Python API is not available. Activate the 'openmc' conda environment "
            "before running this mesh workflow."
        ) from exc
    return gmsh


@dataclass(frozen=True)
class FuelRingDefinition:
    index: int
    r_min_m: float
    r_max_m: float
    z_min_m: float
    z_max_m: float

    @property
    def region_name(self) -> str:
        return f"core_fuel_ring_{self.index}"


@dataclass(frozen=True)
class ReactorGeometryDefinition:
    fuel_element_radius_m: float
    fuel_element_height_m: float
    moderator_radius_m: float
    moderator_height_m: float
    reflector_radius_m: float
    reflector_height_m: float
    rod_radius_m: float
    parked_rod_z_min_m: float
    parked_rod_z_max_m: float
    core_radius_m: float
    core_height_m: float
    central_channel_radius_m: float
    fuel_rings: tuple[FuelRingDefinition, ...]

    @property
    def fuel_element_z_min_m(self) -> float:
        return -0.5 * self.fuel_element_height_m

    @property
    def fuel_element_z_max_m(self) -> float:
        return 0.5 * self.fuel_element_height_m

    @property
    def moderator_z_min_m(self) -> float:
        return -0.5 * self.moderator_height_m

    @property
    def moderator_z_max_m(self) -> float:
        return 0.5 * self.moderator_height_m

    @property
    def reflector_z_min_m(self) -> float:
        return -0.5 * self.reflector_height_m

    @property
    def reflector_z_max_m(self) -> float:
        return 0.5 * self.reflector_height_m

    @property
    def core_z_min_m(self) -> float:
        return -0.5 * self.core_height_m

    @property
    def core_z_max_m(self) -> float:
        return 0.5 * self.core_height_m

    def region_names(self) -> list[str]:
        return [
            "core_central_moderator_channel",
            *(ring.region_name for ring in self.fuel_rings),
            "core_control_rod",
            "core_heavy_water_coolant_and_moderator",
            "moderator",
            "reflector",
        ]


@dataclass(frozen=True)
class MeshSizingDefinition:
    reflector_size_m: float = 2.0
    moderator_size_m: float = 2.0
    core_size_m: float = 0.5
    fuel_size_m: float = 0.2
    rod_size_m: float = 0.2
    axis_epsilon_m: float = 1.0e-4
    wedge_angle_deg: float = 1.0
    experimental_theta_divisions: int = 32
    optimize: bool = False
    size_from_curvature: bool = False
    size_from_boundary: bool = False
    size_from_points: bool = False


@dataclass(frozen=True)
class StructuredBlock:
    region: str
    r_min_m: float
    r_max_m: float
    z_min_m: float
    z_max_m: float

    @property
    def area_m2(self) -> float:
        return (self.r_max_m - self.r_min_m) * (self.z_max_m - self.z_min_m)


@dataclass(frozen=True)
class StructuredGrid:
    radial_edges_m: tuple[float, ...]
    axial_edges_m: tuple[float, ...]
    occupied_cells: dict[tuple[int, int], str]
    blocks: tuple[StructuredBlock, ...]


def default_reactor_geometry() -> ReactorGeometryDefinition:
    return ReactorGeometryDefinition(
        fuel_element_radius_m=0.5,
        fuel_element_height_m=4.0,
        moderator_radius_m=2.5,
        moderator_height_m=6.0,
        reflector_radius_m=5.0,
        reflector_height_m=10.0,
        rod_radius_m=0.04,
        parked_rod_z_min_m=1.5,
        parked_rod_z_max_m=2.0,
        core_radius_m=0.5,
        core_height_m=3.0,
        central_channel_radius_m=0.04,
        fuel_rings=(
            FuelRingDefinition(1, 0.045, 0.05, -1.5, 1.5),
            FuelRingDefinition(2, 0.12, 0.125, -1.5, 1.5),
            FuelRingDefinition(3, 0.195, 0.2, -1.5, 1.5),
            FuelRingDefinition(4, 0.27, 0.275, -1.5, 1.5),
            FuelRingDefinition(5, 0.345, 0.35, -1.5, 1.5),
            FuelRingDefinition(6, 0.42, 0.425, -1.5, 1.5),
            FuelRingDefinition(7, 0.495, 0.5, -1.5, 1.5),
        ),
    )


def geometry_payload(geometry: ReactorGeometryDefinition | None = None) -> dict[str, Any]:
    selected = geometry or default_reactor_geometry()
    payload = asdict(selected)
    payload["fuel_rings"] = [
        {
            "cell_name": f"fuel_ring_{ring.index}",
            "r_min_m": ring.r_min_m,
            "r_max_m": ring.r_max_m,
            "z_min_m": ring.z_min_m,
            "z_max_m": ring.z_max_m,
        }
        for ring in selected.fuel_rings
    ]
    return payload


def mesh_region_mapping(geometry: ReactorGeometryDefinition | None = None) -> list[dict[str, str]]:
    selected = geometry or default_reactor_geometry()
    return [{"region": name, "cell_zone": name} for name in selected.region_names()]


def _region_target_size(region_name: str, sizing: MeshSizingDefinition) -> float:
    if region_name.startswith("core_fuel_ring_"):
        return sizing.fuel_size_m
    if region_name == "core_control_rod":
        return sizing.rod_size_m
    if region_name in {"core_central_moderator_channel", "core_heavy_water_coolant_and_moderator"}:
        return sizing.core_size_m
    if region_name == "moderator":
        return sizing.moderator_size_m
    if region_name == "reflector":
        return sizing.reflector_size_m
    raise ValueError(f"Unsupported region name: {region_name}")


def _material_radial_edges(geometry: ReactorGeometryDefinition) -> list[float]:
    return sorted({
        0.0,
        geometry.central_channel_radius_m,
        geometry.fuel_element_radius_m,
        geometry.moderator_radius_m,
        geometry.reflector_radius_m,
        *(ring.r_min_m for ring in geometry.fuel_rings),
        *(ring.r_max_m for ring in geometry.fuel_rings),
    })


def _material_axial_edges(geometry: ReactorGeometryDefinition) -> list[float]:
    return sorted({
        geometry.reflector_z_min_m,
        geometry.moderator_z_min_m,
        geometry.fuel_element_z_min_m,
        geometry.core_z_min_m,
        geometry.core_z_max_m,
        geometry.fuel_element_z_max_m,
        geometry.moderator_z_max_m,
        geometry.reflector_z_max_m,
    })


def _classify_region(geometry: ReactorGeometryDefinition, radius_m: float, z_m: float) -> str | None:
    if geometry.fuel_element_z_min_m <= z_m <= geometry.core_z_max_m and radius_m <= geometry.central_channel_radius_m:
        return "core_central_moderator_channel"
    if geometry.parked_rod_z_min_m <= z_m <= geometry.parked_rod_z_max_m and radius_m <= geometry.rod_radius_m:
        return "core_control_rod"
    for ring in geometry.fuel_rings:
        if ring.z_min_m <= z_m <= ring.z_max_m and ring.r_min_m <= radius_m <= ring.r_max_m:
            return ring.region_name
    if geometry.fuel_element_z_min_m <= z_m <= geometry.fuel_element_z_max_m and radius_m <= geometry.fuel_element_radius_m:
        return "core_heavy_water_coolant_and_moderator"
    if geometry.moderator_z_min_m <= z_m <= geometry.moderator_z_max_m and radius_m <= geometry.moderator_radius_m:
        return "moderator"
    if geometry.reflector_z_min_m <= z_m <= geometry.reflector_z_max_m and radius_m <= geometry.reflector_radius_m:
        return "reflector"
    return None


def _subdivide_interval(start_m: float, end_m: float, target_size_m: float) -> list[float]:
    if target_size_m <= 0.0:
        raise ValueError(f"Target mesh size must be positive, got {target_size_m}")
    length_m = end_m - start_m
    if length_m <= 0.0:
        raise ValueError(f"Mesh interval must be positive, got [{start_m}, {end_m}]")
    cell_count = max(1, math.ceil(length_m / target_size_m))
    step_m = length_m / cell_count
    return [start_m + step_m * index for index in range(cell_count)] + [end_m]


def _merge_subdivided_edges(material_edges: list[float], interval_targets: dict[int, float]) -> tuple[float, ...]:
    merged: list[float] = []
    for interval_index, (start_m, end_m) in enumerate(zip(material_edges[:-1], material_edges[1:], strict=True)):
        interval_edges = _subdivide_interval(start_m, end_m, interval_targets[interval_index])
        if not merged:
            merged.extend(interval_edges)
        else:
            merged.extend(interval_edges[1:])
    return tuple(round(edge, 12) for edge in merged)


def build_structured_grid(
    geometry: ReactorGeometryDefinition | None = None,
    sizing: MeshSizingDefinition | None = None,
) -> StructuredGrid:
    selected_geometry = geometry or default_reactor_geometry()
    selected_sizing = sizing or MeshSizingDefinition()
    if not 0.0 < selected_sizing.axis_epsilon_m < selected_geometry.central_channel_radius_m:
        raise ValueError("axis_epsilon_m must be positive and smaller than the central moderator channel radius")

    material_radial_edges = _material_radial_edges(selected_geometry)
    material_radial_edges[0] = round(selected_sizing.axis_epsilon_m, 12)
    material_axial_edges = _material_axial_edges(selected_geometry)
    coarse_cells: dict[tuple[int, int], str] = {}
    for radial_index, (r_min_m, r_max_m) in enumerate(zip(material_radial_edges[:-1], material_radial_edges[1:], strict=True)):
        for axial_index, (z_min_m, z_max_m) in enumerate(zip(material_axial_edges[:-1], material_axial_edges[1:], strict=True)):
            region_name = _classify_region(selected_geometry, 0.5 * (r_min_m + r_max_m), 0.5 * (z_min_m + z_max_m))
            if region_name is not None:
                coarse_cells[(radial_index, axial_index)] = region_name

    if not coarse_cells:
        raise ValueError("No occupied reactor cells were classified from the manual geometry definition")

    radial_targets = {
        radial_index: min(
            _region_target_size(region_name, selected_sizing)
            for (candidate_index, _), region_name in coarse_cells.items()
            if candidate_index == radial_index
        )
        for radial_index in range(len(material_radial_edges) - 1)
    }
    axial_targets = {
        axial_index: min(
            _region_target_size(region_name, selected_sizing)
            for (_, candidate_index), region_name in coarse_cells.items()
            if candidate_index == axial_index
        )
        for axial_index in range(len(material_axial_edges) - 1)
    }

    radial_edges_m = _merge_subdivided_edges(material_radial_edges, radial_targets)
    axial_edges_m = _merge_subdivided_edges(material_axial_edges, axial_targets)

    occupied_cells: dict[tuple[int, int], str] = {}
    blocks: list[StructuredBlock] = []
    for radial_index, (r_min_m, r_max_m) in enumerate(zip(radial_edges_m[:-1], radial_edges_m[1:], strict=True)):
        for axial_index, (z_min_m, z_max_m) in enumerate(zip(axial_edges_m[:-1], axial_edges_m[1:], strict=True)):
            region_name = _classify_region(selected_geometry, 0.5 * (r_min_m + r_max_m), 0.5 * (z_min_m + z_max_m))
            if region_name is None:
                continue
            occupied_cells[(radial_index, axial_index)] = region_name
            blocks.append(StructuredBlock(region_name, r_min_m, r_max_m, z_min_m, z_max_m))

    return StructuredGrid(
        radial_edges_m=radial_edges_m,
        axial_edges_m=axial_edges_m,
        occupied_cells=occupied_cells,
        blocks=tuple(blocks),
    )


def mesh_metadata(
    mesh_file: Path = DEFAULT_MESH_FILE,
    manifest_file: Path = DEFAULT_MANIFEST_FILE,
    geometry: ReactorGeometryDefinition | None = None,
    sizing: MeshSizingDefinition | None = None,
    mesh_kind: str = WEDGE_MESH_KIND,
    structured_grid: StructuredGrid | None = None,
) -> dict[str, Any]:
    selected_geometry = geometry or default_reactor_geometry()
    selected_sizing = sizing or MeshSizingDefinition()
    selected_grid = structured_grid or build_structured_grid(selected_geometry, selected_sizing)
    region_mapping = mesh_region_mapping(selected_geometry)

    if mesh_kind == WEDGE_MESH_KIND:
        expected_boundary_patches = ["wedge_front", "wedge_back", "axis", "bottom", "top", "outer"]
        boundary_patch_types = {
            "wedge_front": "wedge",
            "wedge_back": "wedge",
            "axis": "symmetryPlane",
            "bottom": "patch",
            "top": "patch",
            "outer": "patch",
        }
    elif mesh_kind == FULL_3D_EXPERIMENTAL_MESH_KIND:
        expected_boundary_patches = ["walls"]
        boundary_patch_types = {"walls": "patch"}
    else:
        raise ValueError(f"Unsupported mesh kind {mesh_kind!r}")

    return {
        "kind": mesh_kind,
        "mesh_file": str(mesh_file),
        "mesh_file_name": mesh_file.name,
        "manifest_file": str(manifest_file),
        "manifest_file_name": manifest_file.name,
        "geometry_source": "manual",
        "expected_cell_zones": [item["cell_zone"] for item in region_mapping],
        "expected_boundary_patches": expected_boundary_patches,
        "boundary_patch_types": boundary_patch_types,
        "region_to_cell_zone": region_mapping,
        "geometry": geometry_payload(selected_geometry),
        "mesh_sizing": asdict(selected_sizing),
        "axis_representation": "thin_inner_cylinder" if mesh_kind == WEDGE_MESH_KIND else "full_volume",
        "radial_edges_m": list(selected_grid.radial_edges_m),
        "axial_edges_m": list(selected_grid.axial_edges_m),
        "blocks": [
            {
                "block_id": f"cell_{block_index:05d}",
                "region": block.region,
                "r_min_m": block.r_min_m,
                "r_max_m": block.r_max_m,
                "z_min_m": block.z_min_m,
                "z_max_m": block.z_max_m,
                "area_m2": block.area_m2,
            }
            for block_index, block in enumerate(selected_grid.blocks)
        ],
    }


def _interval_node_count(length_m: float, target_size_m: float) -> int:
    if target_size_m <= 0.0:
        raise ValueError(f"Target mesh size must be positive, got {target_size_m}")
    element_count = max(1, math.ceil(length_m / target_size_m))
    return element_count + 1


def _build_planar_surfaces(
    gmsh: Any,
    grid: StructuredGrid,
    sizing: MeshSizingDefinition,
) -> list[tuple[int, str, bool]]:
    radial_edges = grid.radial_edges_m
    axial_edges = grid.axial_edges_m
    occupied_cells = grid.occupied_cells

    radial_targets = {
        radial_index: min(
            _region_target_size(region_name, sizing)
            for (candidate_index, _), region_name in occupied_cells.items()
            if candidate_index == radial_index
        )
        for radial_index in range(len(radial_edges) - 1)
    }
    axial_targets = {
        axial_index: min(
            _region_target_size(region_name, sizing)
            for (_, candidate_index), region_name in occupied_cells.items()
            if candidate_index == axial_index
        )
        for axial_index in range(len(axial_edges) - 1)
    }

    geo = gmsh.model.geo
    point_tags: dict[tuple[int, int], int] = {}
    for radial_index, radius_m in enumerate(radial_edges):
        for axial_index, z_m in enumerate(axial_edges):
            point_tags[(radial_index, axial_index)] = geo.addPoint(radius_m, 0.0, z_m)

    horizontal_lines: dict[tuple[int, int], int] = {}
    vertical_lines: dict[tuple[int, int], int] = {}
    for radial_index in range(len(radial_edges) - 1):
        for axial_index in range(len(axial_edges)):
            horizontal_lines[(radial_index, axial_index)] = geo.addLine(
                point_tags[(radial_index, axial_index)],
                point_tags[(radial_index + 1, axial_index)],
            )
    for radial_index in range(len(radial_edges)):
        for axial_index in range(len(axial_edges) - 1):
            vertical_lines[(radial_index, axial_index)] = geo.addLine(
                point_tags[(radial_index, axial_index)],
                point_tags[(radial_index, axial_index + 1)],
            )

    surface_entities: list[tuple[int, str, bool]] = []
    for (radial_index, axial_index), region_name in sorted(occupied_cells.items()):
        curve_loop = geo.addCurveLoop([
            horizontal_lines[(radial_index, axial_index)],
            vertical_lines[(radial_index + 1, axial_index)],
            -horizontal_lines[(radial_index, axial_index + 1)],
            -vertical_lines[(radial_index, axial_index)],
        ])
        surface_entities.append((geo.addPlaneSurface([curve_loop]), region_name, True))

    geo.synchronize()

    for (radial_index, _), line_tag in horizontal_lines.items():
        node_count = _interval_node_count(radial_edges[radial_index + 1] - radial_edges[radial_index], radial_targets[radial_index])
        gmsh.model.mesh.setTransfiniteCurve(line_tag, node_count)
    for (_, axial_index), line_tag in vertical_lines.items():
        node_count = _interval_node_count(axial_edges[axial_index + 1] - axial_edges[axial_index], axial_targets[axial_index])
        gmsh.model.mesh.setTransfiniteCurve(line_tag, node_count)
    for surface_tag, _, recombine_surface in surface_entities:
        if recombine_surface:
            gmsh.model.mesh.setTransfiniteSurface(surface_tag)
            gmsh.model.mesh.setRecombine(2, surface_tag)
    return surface_entities


def _build_wedge_volumes(
    gmsh: Any,
    grid: StructuredGrid,
    sizing: MeshSizingDefinition,
) -> list[tuple[int, str]]:
    if sizing.wedge_angle_deg <= 0.0 or sizing.wedge_angle_deg >= 15.0:
        raise ValueError("wedge_angle_deg must be greater than 0 and comfortably smaller than 15 degrees")

    surface_entities = _build_planar_surfaces(gmsh, grid, sizing)
    geo = gmsh.model.geo
    wedge_angle_rad = math.radians(sizing.wedge_angle_deg)
    half_wedge_angle_rad = 0.5 * wedge_angle_rad
    volume_entities: list[tuple[int, str]] = []
    structured_volume_tags: list[int] = []

    for surface_tag, region_name, recombine_surface in surface_entities:
        copied_surface = geo.copy([(2, surface_tag)])
        geo.rotate(copied_surface, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -half_wedge_angle_rad)
        revolved = geo.revolve(
            copied_surface,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            wedge_angle_rad,
            [1],
            [1.0],
            True,
        )
        volume_tags = [tag for dim, tag in revolved if dim == 3]
        if len(volume_tags) != 1:
            raise ValueError(f"Expected one revolved wedge volume for region {region_name!r}, found {len(volume_tags)}")
        volume_tag = volume_tags[0]
        volume_entities.append((volume_tag, region_name))
        if recombine_surface:
            structured_volume_tags.append(volume_tag)

    geo.synchronize()
    for volume_tag in structured_volume_tags:
        gmsh.model.mesh.setTransfiniteVolume(volume_tag)
    return volume_entities


def _build_full_3d_volumes(
    gmsh: Any,
    grid: StructuredGrid,
    sizing: MeshSizingDefinition,
) -> list[tuple[int, str]]:
    if sizing.experimental_theta_divisions < 4 or sizing.experimental_theta_divisions % 2 != 0:
        raise ValueError("experimental_theta_divisions must be an even integer >= 4")

    surface_entities = _build_planar_surfaces(gmsh, grid, sizing)
    geo = gmsh.model.geo
    half_turn = math.pi
    half_turn_layers = sizing.experimental_theta_divisions // 2
    volume_entities: list[tuple[int, str]] = []
    structured_volume_tags: list[int] = []
    for surface_tag, region_name, recombine_surface in surface_entities:
        created_volume_tags: list[int] = []
        for rotation_angle in (0.0, half_turn):
            copied_surface = geo.copy([(2, surface_tag)])
            if rotation_angle:
                geo.rotate(copied_surface, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, rotation_angle)
            revolved = geo.revolve(
                copied_surface,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                half_turn,
                [half_turn_layers],
                [1.0],
                True,
            )
            volume_tags = [tag for dim, tag in revolved if dim == 3]
            if len(volume_tags) > 1:
                raise ValueError(
                    f"Expected at most one revolved volume per half-turn for region {region_name!r}, found {len(volume_tags)}"
                )
            created_volume_tags.extend(volume_tags)
        if not created_volume_tags:
            raise ValueError(f"Structured revolve did not produce any 3D volume for region {region_name!r}")
        volume_entities.extend((volume_tag, region_name) for volume_tag in created_volume_tags)
        if recombine_surface:
            structured_volume_tags.extend(created_volume_tags)

    geo.synchronize()
    for volume_tag in structured_volume_tags:
        gmsh.model.mesh.setTransfiniteVolume(volume_tag)
    return volume_entities


def _wedge_boundary_name(
    gmsh: Any,
    geometry: ReactorGeometryDefinition,
    surface_tag: int,
    axis_radius_m: float,
    wedge_angle_deg: float,
) -> str:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surface_tag)
    z_tol = 1.0e-8
    if abs(zmin - zmax) <= z_tol and abs(zmin - geometry.reflector_z_min_m) <= z_tol:
        return "bottom"
    if abs(zmin - zmax) <= z_tol and abs(zmax - geometry.reflector_z_max_m) <= z_tol:
        return "top"

    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    angle_rad = math.atan2(center_y, center_x) if (center_x or center_y) else 0.0
    half_wedge_angle_rad = math.radians(0.5 * wedge_angle_deg)
    angle_tol = max(1.0e-6, abs(half_wedge_angle_rad) * 0.1)
    if abs(angle_rad + half_wedge_angle_rad) <= angle_tol:
        return "wedge_front"
    if abs(angle_rad - half_wedge_angle_rad) <= angle_tol:
        return "wedge_back"
    corner_radii = [
        math.hypot(x_coord, y_coord)
        for x_coord in (xmin, xmax)
        for y_coord in (ymin, ymax)
    ]
    radial_tol = max(1.0e-8, axis_radius_m * 0.25)
    if max(abs(radius - axis_radius_m) for radius in corner_radii) <= radial_tol:
        return "axis"
    return "outer"


def _add_physical_groups(
    gmsh: Any,
    geometry: ReactorGeometryDefinition,
    volume_entities: list[tuple[int, str]],
    mesh_kind: str,
    sizing: MeshSizingDefinition,
    axis_radius_m: float,
) -> dict[str, list[str]]:
    grouped_volumes: dict[str, list[int]] = {}
    for tag, name in volume_entities:
        grouped_volumes.setdefault(name, []).append(tag)

    physical_volume_names: list[str] = []
    for name in geometry.region_names():
        tags = grouped_volumes.get(name, [])
        if not tags:
            raise ValueError(f"Missing fragmented volume tags for region {name!r}")
        group_tag = gmsh.model.addPhysicalGroup(3, tags)
        gmsh.model.setPhysicalName(3, group_tag, name)
        physical_volume_names.append(name)

    boundary_surfaces = [
        tag
        for dim, tag in gmsh.model.getBoundary(
            [(3, tag) for tag, _ in volume_entities],
            combined=True,
            oriented=False,
            recursive=False,
        )
        if dim == 2
    ]
    if not boundary_surfaces:
        raise ValueError("Could not resolve any external boundary surfaces for the reactor mesh")

    if mesh_kind == FULL_3D_EXPERIMENTAL_MESH_KIND:
        wall_group = gmsh.model.addPhysicalGroup(2, sorted(set(boundary_surfaces)))
        gmsh.model.setPhysicalName(2, wall_group, "walls")
        return {
            "physical_volumes": physical_volume_names,
            "physical_boundaries": ["walls"],
        }

    if mesh_kind != WEDGE_MESH_KIND:
        raise ValueError(f"Unsupported mesh kind {mesh_kind!r}")

    grouped_surfaces: dict[str, list[int]] = {}
    for surface_tag in sorted(set(boundary_surfaces)):
        boundary_name = _wedge_boundary_name(gmsh, geometry, surface_tag, axis_radius_m, sizing.wedge_angle_deg)
        grouped_surfaces.setdefault(boundary_name, []).append(surface_tag)

    expected_boundary_names = ["wedge_front", "wedge_back", "axis", "bottom", "top", "outer"]
    for boundary_name in expected_boundary_names:
        tags = grouped_surfaces.get(boundary_name, [])
        if not tags:
            raise ValueError(f"Missing wedge boundary surfaces for patch {boundary_name!r}")
        group_tag = gmsh.model.addPhysicalGroup(2, tags)
        gmsh.model.setPhysicalName(2, group_tag, boundary_name)

    return {
        "physical_volumes": physical_volume_names,
        "physical_boundaries": expected_boundary_names,
    }


def _apply_mesh_fields(gmsh: Any, sizing: MeshSizingDefinition) -> None:
    gmsh.option.setNumber("Mesh.MeshSizeMin", min(sizing.rod_size_m, sizing.fuel_size_m))
    gmsh.option.setNumber("Mesh.MeshSizeMax", sizing.reflector_size_m)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.Optimize", 1 if sizing.optimize else 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1 if sizing.size_from_curvature else 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1 if sizing.size_from_boundary else 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1 if sizing.size_from_points else 0)


def write_mesh_assets(
    mesh_file: Path = DEFAULT_MESH_FILE,
    manifest_file: Path = DEFAULT_MANIFEST_FILE,
    geometry: ReactorGeometryDefinition | None = None,
    sizing: MeshSizingDefinition | None = None,
    mesh_kind: str = WEDGE_MESH_KIND,
) -> dict[str, Any]:
    gmsh = _import_gmsh()
    selected_geometry = geometry or default_reactor_geometry()
    selected_sizing = sizing or MeshSizingDefinition()
    structured_grid = build_structured_grid(selected_geometry, selected_sizing)

    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("concentric_reactor_mesh")
        if mesh_kind == WEDGE_MESH_KIND:
            volume_entities = _build_wedge_volumes(gmsh, structured_grid, selected_sizing)
        elif mesh_kind == FULL_3D_EXPERIMENTAL_MESH_KIND:
            volume_entities = _build_full_3d_volumes(gmsh, structured_grid, selected_sizing)
        else:
            raise ValueError(f"Unsupported mesh kind {mesh_kind!r}")
        physical_groups = _add_physical_groups(
            gmsh,
            selected_geometry,
            volume_entities,
            mesh_kind,
            selected_sizing,
            axis_radius_m=structured_grid.radial_edges_m[0],
        )
        _apply_mesh_fields(gmsh, selected_sizing)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(mesh_file))
    finally:
        gmsh.finalize()

    manifest = mesh_metadata(
        mesh_file=mesh_file,
        manifest_file=manifest_file,
        geometry=selected_geometry,
        sizing=selected_sizing,
        mesh_kind=mesh_kind,
        structured_grid=structured_grid,
    )
    manifest.update(physical_groups)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_foam_names(path: Path, *, object_name: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find required OpenFOAM file: {path}")
    text = path.read_text(encoding="utf-8")
    if object_name == "cellZones":
        meta_match = re.search(r"names\s*\(\s*([^)]+?)\s*\)\s*;", text, re.DOTALL)
        if meta_match:
            return [token for token in meta_match.group(1).split() if token]
    return [name for name in re.findall(r"^\s*([A-Za-z0-9_]+)\s*\n\s*\{", text, re.MULTILINE) if name != "FoamFile"]


def _parse_boundary_patch_types(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find required OpenFOAM file: {path}")
    text = path.read_text(encoding="utf-8")
    patch_types: dict[str, str] = {}
    for match in _BOUNDARY_BLOCK_PATTERN.finditer(text):
        name = match.group("name")
        type_match = re.search(r"^\s*type\s+([A-Za-z0-9_]+)\s*;", match.group("body"), re.MULTILINE)
        if type_match is not None:
            patch_types[name] = type_match.group(1)
    return patch_types


def _rewrite_boundary_patch(text: str, patch_name: str, patch_type: str) -> str:
    pattern = re.compile(
        rf"(^\s*{re.escape(patch_name)}\s*\n\s*\{{)(?P<body>.*?)(^\s*\}})",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not find patch {patch_name!r} in boundary file")

    body = match.group("body")
    if re.search(r"^\s*type\s+", body, re.MULTILINE):
        body = re.sub(r"(^\s*type\s+)[A-Za-z0-9_]+(\s*;)", rf"\1{patch_type}\2", body, flags=re.MULTILINE)
    else:
        body = f"\n        type            {patch_type};" + body

    if patch_type == "wedge":
        if re.search(r"^\s*inGroups\s+", body, re.MULTILINE):
            body = re.sub(r"^\s*inGroups\s+.*?;\s*$", "        inGroups        1(wedge);", body, flags=re.MULTILINE)
        else:
            body = re.sub(
                rf"(^\s*type\s+{patch_type}\s*;\s*$)",
                "\\1\n        inGroups        1(wedge);",
                body,
                flags=re.MULTILINE,
            )
    else:
        body = re.sub(r"^\s*inGroups\s+.*?;\s*$", "", body, flags=re.MULTILINE)

    return text[:match.start("body")] + body + text[match.end("body"):]


def configure_import(case_dir: Path, manifest_file: Path, region: str = "neutroRegion") -> dict[str, Any]:
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    boundary_patch_types = dict(manifest.get("boundary_patch_types", {}))
    boundary_path = case_dir / "constant" / region / "polyMesh" / "boundary"
    boundary_text = boundary_path.read_text(encoding="utf-8")
    for patch_name, patch_type in boundary_patch_types.items():
        boundary_text = _rewrite_boundary_patch(boundary_text, patch_name, patch_type)
    boundary_path.write_text(boundary_text, encoding="utf-8")
    return {
        "region": region,
        "configured_patches": boundary_patch_types,
    }


def validate_import(case_dir: Path, manifest_file: Path, region: str = "neutroRegion") -> dict[str, Any]:
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    poly_mesh_dir = case_dir / "constant" / region / "polyMesh"
    cell_zone_names = _parse_foam_names(poly_mesh_dir / "cellZones", object_name="cellZones")
    boundary_names = _parse_foam_names(poly_mesh_dir / "boundary", object_name="boundary")
    boundary_types = _parse_boundary_patch_types(poly_mesh_dir / "boundary")

    expected_zones = list(manifest["expected_cell_zones"])
    expected_patches = list(manifest["expected_boundary_patches"])
    expected_patch_types = dict(manifest.get("boundary_patch_types", {}))
    missing_zones = [name for name in expected_zones if name not in cell_zone_names]
    missing_patches = [name for name in expected_patches if name not in boundary_names]
    unexpected_default_faces = "defaultFaces" in boundary_names
    wrong_patch_types = {
        name: {
            "expected": expected_patch_types[name],
            "found": boundary_types.get(name),
        }
        for name in expected_patch_types
        if boundary_types.get(name) != expected_patch_types[name]
    }
    if missing_zones or missing_patches or wrong_patch_types or unexpected_default_faces:
        details = {
            "missing_cell_zones": missing_zones,
            "missing_boundary_patches": missing_patches,
            "wrong_boundary_patch_types": wrong_patch_types,
            "unexpected_defaultFaces": unexpected_default_faces,
            "found_cell_zones": cell_zone_names,
            "found_boundary_patches": boundary_names,
            "found_boundary_patch_types": boundary_types,
        }
        raise RuntimeError(json.dumps(details, indent=2))
    return {
        "region": region,
        "found_cell_zones": cell_zone_names,
        "found_boundary_patches": boundary_names,
        "found_boundary_patch_types": boundary_types,
        "expected_cell_zones": expected_zones,
        "expected_boundary_patches": expected_patches,
        "expected_boundary_patch_types": expected_patch_types,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate the concentric reactor Gmsh mesh for GeN-Foam.")
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="Generate the reactor .msh file and JSON manifest")
    generate.add_argument("--mesh-file", type=Path, default=DEFAULT_MESH_FILE)
    generate.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)
    generate.add_argument("--mesh-kind", choices=[WEDGE_MESH_KIND, FULL_3D_EXPERIMENTAL_MESH_KIND], default=WEDGE_MESH_KIND)
    generate.add_argument("--reflector-size-m", type=float, default=MeshSizingDefinition.reflector_size_m)
    generate.add_argument("--moderator-size-m", type=float, default=MeshSizingDefinition.moderator_size_m)
    generate.add_argument("--core-size-m", type=float, default=MeshSizingDefinition.core_size_m)
    generate.add_argument("--fuel-size-m", type=float, default=MeshSizingDefinition.fuel_size_m)
    generate.add_argument("--rod-size-m", type=float, default=MeshSizingDefinition.rod_size_m)
    generate.add_argument("--wedge-angle-deg", type=float, default=MeshSizingDefinition.wedge_angle_deg)
    generate.add_argument("--experimental-theta-divisions", type=int, default=MeshSizingDefinition.experimental_theta_divisions)
    generate.add_argument("--optimize", action="store_true", help="Enable Gmsh mesh optimization after generation")
    generate.add_argument(
        "--size-from-curvature",
        action="store_true",
        help="Allow Gmsh curvature-based local refinement in addition to the background size field",
    )
    generate.add_argument(
        "--size-from-boundary",
        action="store_true",
        help="Allow boundary sizes to extend into adjacent volumes during meshing",
    )
    generate.add_argument(
        "--size-from-points",
        action="store_true",
        help="Allow Gmsh point sizes to influence mesh density",
    )

    configure = subparsers.add_parser("configure-import", help="Apply OpenFOAM patch types to an imported mesh")
    configure.add_argument("--case-dir", type=Path, default=CASE_DIR)
    configure.add_argument("--region", default="neutroRegion")
    configure.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)

    validate = subparsers.add_parser("validate-import", help="Validate imported OpenFOAM patches and cell zones against the manifest")
    validate.add_argument("--case-dir", type=Path, default=CASE_DIR)
    validate.add_argument("--region", default="neutroRegion")
    validate.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)

    parser.set_defaults(command="generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        sizing = MeshSizingDefinition(
            reflector_size_m=args.reflector_size_m,
            moderator_size_m=args.moderator_size_m,
            core_size_m=args.core_size_m,
            fuel_size_m=args.fuel_size_m,
            rod_size_m=args.rod_size_m,
            wedge_angle_deg=args.wedge_angle_deg,
            experimental_theta_divisions=args.experimental_theta_divisions,
            optimize=args.optimize,
            size_from_curvature=args.size_from_curvature,
            size_from_boundary=args.size_from_boundary,
            size_from_points=args.size_from_points,
        )
        manifest = write_mesh_assets(
            mesh_file=args.mesh_file,
            manifest_file=args.manifest_file,
            sizing=sizing,
            mesh_kind=args.mesh_kind,
        )
        print(json.dumps({"mesh_file": manifest["mesh_file"], "manifest_file": manifest["manifest_file"], "kind": manifest["kind"]}, indent=2))
        return
    if args.command == "configure-import":
        result = configure_import(case_dir=args.case_dir.resolve(), manifest_file=args.manifest_file.resolve(), region=args.region)
        print(json.dumps(result, indent=2))
        return
    if args.command == "validate-import":
        result = validate_import(case_dir=args.case_dir.resolve(), manifest_file=args.manifest_file.resolve(), region=args.region)
        print(json.dumps(result, indent=2))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
