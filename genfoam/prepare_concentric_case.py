from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gmsh_reactor_mesh import (
    WEDGE_MESH_KIND,
    default_reactor_geometry,
    geometry_payload as manual_geometry_payload,
    mesh_metadata as gmsh_mesh_metadata,
    mesh_region_mapping,
    write_mesh_assets,
)


CASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CASE_DIR.parent
DEFAULT_MGXS_EXPORT_DIR = PROJECT_DIR / "openmc" / "build" / "concentric" / "mgxs_export"
DEFAULT_OUTPUT_DIR = CASE_DIR / "constant" / "generated"
_FUEL_RING_LABEL_PATTERN = re.compile(r"^core_fuel_ring_(\d+)$")


@dataclass(frozen=True)
class OverlayZone:
    label: str
    region: str
    r_min_m: float
    r_max_m: float
    z_min_m: float
    z_max_m: float

    def contains(self, r_m: float, z_m: float) -> bool:
        return self.r_min_m <= r_m < self.r_max_m and self.z_min_m <= z_m < self.z_max_m


@dataclass(frozen=True)
class MeshBlock:
    block_id: str
    region: str
    r_min_m: float
    r_max_m: float
    z_min_m: float
    z_max_m: float

    @property
    def area_m2(self) -> float:
        return (self.r_max_m - self.r_min_m) * (self.z_max_m - self.z_min_m)


def require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Could not find required directory: {path}")
    return path.resolve()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Could not find required file: {path}")
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_surface_value(surface: ET.Element) -> float:
    coeffs = [float(token) for token in surface.attrib["coeffs"].split()]
    surface_type = surface.attrib["type"]
    if surface_type == "z-cylinder":
        return coeffs[-1]
    if surface_type == "z-plane":
        return coeffs[0]
    raise ValueError(f"Unsupported surface type in model.xml: {surface_type}")


def extract_cell_bounds(root: ET.Element, cell_name: str) -> dict[str, float | list[float]]:
    cell = root.find(f".//cell[@name='{cell_name}']")
    if cell is None:
        raise ValueError(f"Could not find cell named {cell_name!r} in model.xml")

    surface_map = {surface.attrib["id"]: surface for surface in root.findall(".//surface")}
    surface_ids = {token.lstrip("-") for token in re.findall(r"-?\d+", cell.attrib["region"])}
    surfaces = [surface_map[surface_id] for surface_id in surface_ids]

    cylinder_radii = sorted(
        parse_surface_value(surface)
        for surface in surfaces
        if surface.attrib["type"] == "z-cylinder"
    )
    z_planes = sorted(
        parse_surface_value(surface)
        for surface in surfaces
        if surface.attrib["type"] == "z-plane"
    )
    if not z_planes:
        raise ValueError(f"Cell {cell_name!r} is missing axial surfaces")

    return {
        "radii_cm": cylinder_radii,
        "r_min_cm": cylinder_radii[0] if len(cylinder_radii) > 1 else 0.0,
        "r_max_cm": cylinder_radii[-1] if cylinder_radii else 0.0,
        "z_min_cm": z_planes[0],
        "z_max_cm": z_planes[-1],
        "height_cm": z_planes[-1] - z_planes[0],
    }


def _cm_to_m(value_cm: float) -> float:
    return float(value_cm) * 1.0e-2


def extract_concentric_geometry(model_xml_path: Path) -> dict[str, Any]:
    root = ET.parse(model_xml_path).getroot()
    fuel_element = extract_cell_bounds(root, "fuel_element")
    moderator = extract_cell_bounds(root, "d2o_tank")
    reflector = extract_cell_bounds(root, "h2o_tank")
    control_rod = extract_cell_bounds(root, "control_rod")

    geometry: dict[str, Any] = {
        "fuel_element_radius_m": _cm_to_m(float(fuel_element["r_max_cm"])),
        "fuel_element_height_m": _cm_to_m(float(fuel_element["height_cm"])),
        "moderator_radius_m": _cm_to_m(float(moderator["r_max_cm"])),
        "moderator_height_m": _cm_to_m(float(moderator["height_cm"])),
        "reflector_radius_m": _cm_to_m(float(reflector["r_max_cm"])),
        "reflector_height_m": _cm_to_m(float(reflector["height_cm"])),
        "outer_height_m": _cm_to_m(float(max(moderator["height_cm"], reflector["height_cm"]))),
        "rod_radius_m": _cm_to_m(float(control_rod["r_max_cm"])),
        "parked_rod_z_min_m": _cm_to_m(float(control_rod["z_min_cm"])),
        "parked_rod_z_max_m": _cm_to_m(float(control_rod["z_max_cm"])),
    }

    fuel_ring_names = sorted(
        (
            cell.attrib["name"]
            for cell in root.findall(".//cell")
            if "name" in cell.attrib and re.match(r"^fuel_ring_\d+$", cell.attrib["name"])
        ),
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    if fuel_ring_names:
        fuel_rings = [extract_cell_bounds(root, name) for name in fuel_ring_names]
        central_channel = extract_cell_bounds(root, "central_moderator_channel")
        geometry["core_radius_m"] = _cm_to_m(float(fuel_element["r_max_cm"]))
        geometry["core_height_m"] = _cm_to_m(float(fuel_rings[0]["height_cm"]))
        geometry["central_channel_radius_m"] = _cm_to_m(float(central_channel["r_max_cm"]))
        geometry["fuel_rings"] = [
            {
                "cell_name": name,
                "r_min_m": _cm_to_m(float(bounds["r_min_cm"])),
                "r_max_m": _cm_to_m(float(bounds["r_max_cm"])),
                "z_min_m": _cm_to_m(float(bounds["z_min_cm"])),
                "z_max_m": _cm_to_m(float(bounds["z_max_cm"])),
            }
            for name, bounds in zip(fuel_ring_names, fuel_rings, strict=True)
        ]
    else:
        geometry["core_radius_m"] = _cm_to_m(float(fuel_element["r_max_cm"]))
        geometry["core_height_m"] = _cm_to_m(float(fuel_element["height_cm"]))
        geometry["central_channel_radius_m"] = _cm_to_m(float(control_rod["r_max_cm"]))
        geometry["fuel_rings"] = []

    return geometry


def sorted_resolved_fuel_ring_labels(export: dict[str, Any]) -> list[str]:
    labels: list[tuple[int, str]] = []
    for label in export["domains"]:
        match = _FUEL_RING_LABEL_PATTERN.match(label)
        if match is None:
            continue
        labels.append((int(match.group(1)), label))
    return [label for _, label in sorted(labels)]


def _collapse_legendre_moment(values: Any) -> Any:
    if not isinstance(values, list):
        return values
    if not values:
        return values
    if isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        if len(values) == len(values[0]) and len(values[0][0]) < len(values):
            return [[moment_values[0] for moment_values in row] for row in values]
        return values[0]
    return values


def _deep_copy(values: Any) -> Any:
    if isinstance(values, list):
        return [_deep_copy(value) for value in values]
    if isinstance(values, dict):
        return {key: _deep_copy(value) for key, value in values.items()}
    return values


def _deep_scale(values: Any, factor: float) -> Any:
    if isinstance(values, list):
        return [_deep_scale(value, factor) for value in values]
    return float(values) * factor


def _group_count(group_constants: dict[str, Any]) -> int:
    diffusion = group_constants["diffusion-coefficient"]["mean"]
    return len(diffusion)


def sanitize_diffusion_coefficients(
    domain_label: str,
    group_constants: dict[str, Any],
    max_diffusion_m: float = 1.0e4,
) -> list[float]:
    diffusion_m = [float(value) * 1.0e-2 for value in group_constants["diffusion-coefficient"]["mean"]]
    absorption_m = [float(value) * 100.0 for value in group_constants["absorption"]["mean"]]
    nu_fission_m = [float(value) * 100.0 for value in group_constants["nu-fission"]["mean"]]
    chi = [float(value) for value in group_constants["chi"]["mean"]]
    scatter_matrix = _collapse_legendre_moment(group_constants["scatter matrix"]["mean"])

    for group_index, value_m in enumerate(diffusion_m):
        needs_fix = not math.isfinite(value_m) or value_m <= 0.0 or value_m > max_diffusion_m
        if not needs_fix:
            continue

        scatter_row = [float(value) for value in scatter_matrix[group_index]]
        scatter_col = [float(row[group_index]) for row in scatter_matrix]
        inactive_group = (
            absorption_m[group_index] == 0.0
            and nu_fission_m[group_index] == 0.0
            and chi[group_index] == 0.0
            and all(value == 0.0 for value in scatter_row)
            and all(value == 0.0 for value in scatter_col)
        )
        if not inactive_group:
            raise ValueError(
                f"Domain {domain_label!r} has an invalid diffusion coefficient in active group {group_index + 1}"
            )

        replacement = next(
            (
                candidate
                for other_index, candidate in enumerate(diffusion_m)
                if other_index != group_index and math.isfinite(candidate) and 0.0 < candidate <= max_diffusion_m
            ),
            1.0,
        )
        diffusion_m[group_index] = replacement
    return diffusion_m


def sanitize_group_vector(values: list[float], fallback: float = 1.0e-6) -> list[float]:
    positive_values = [value for value in values if value > 0.0 and math.isfinite(value)]
    replacement = min(positive_values) if positive_values else fallback
    return [value if value > 0.0 and math.isfinite(value) else replacement for value in values]


def convert_group_constants_to_si(domain_label: str, group_constants: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    diffusion_override_m = sanitize_diffusion_coefficients(domain_label, group_constants)
    inverse_velocity_override = sanitize_group_vector(
        [float(value) * 100.0 for value in group_constants["inverse-velocity"]["mean"]]
    )
    conversion_factors = {
        "total": 100.0,
        "transport": 100.0,
        "absorption": 100.0,
        "nu-fission": 100.0,
        "kappa-fission": 100.0,
        "inverse-velocity": 100.0,
    }

    for xs_type, payload in group_constants.items():
        mean_value = _collapse_legendre_moment(payload["mean"])
        std_dev_value = _collapse_legendre_moment(payload["std_dev"])
        if xs_type == "diffusion-coefficient":
            converted[xs_type] = {
                "mean": diffusion_override_m,
                "std_dev": _deep_scale(std_dev_value, 1.0e-2),
                "units": "m",
            }
            continue
        if xs_type in {"scatter matrix", "nu-scatter matrix"}:
            converted[xs_type] = {
                "mean": _deep_scale(mean_value, 100.0),
                "std_dev": _deep_scale(std_dev_value, 100.0),
                "units": "1/m",
                "legendre_moment": 0,
            }
            continue
        if xs_type == "chi":
            converted[xs_type] = {
                "mean": _deep_copy(mean_value),
                "std_dev": _deep_copy(std_dev_value),
                "units": "1",
            }
            continue
        if xs_type == "inverse-velocity":
            converted[xs_type] = {
                "mean": inverse_velocity_override,
                "std_dev": _deep_scale(std_dev_value, 100.0),
                "units": "s/m",
            }
            continue

        factor = conversion_factors.get(xs_type)
        if factor is None:
            converted[xs_type] = {
                "mean": _deep_copy(mean_value),
                "std_dev": _deep_copy(std_dev_value),
                "units": "unknown",
            }
            continue
        units = "s/m" if xs_type == "inverse-velocity" else "1/m"
        converted[xs_type] = {
            "mean": _deep_scale(mean_value, factor),
            "std_dev": _deep_scale(std_dev_value, factor),
            "units": units,
        }

    converted["metadata"] = {
        "group_count": _group_count(group_constants),
        "sanitized_diffusion": converted["diffusion-coefficient"]["mean"]
        != [float(value) * 1.0e-2 for value in group_constants["diffusion-coefficient"]["mean"]],
    }
    return converted


def build_delayed_neutron_payload(export: dict[str, Any], domain_label: str) -> dict[str, Any]:
    domain_payload = export["domains"][domain_label]["delayed_neutrons"]
    delayed_rows = [
        row
        for row in export.get("delayed_neutron_rows", [])
        if row["domain"] == domain_label
    ]
    return {
        "summary": _deep_copy(domain_payload),
        "rows": delayed_rows,
    }


def build_material_payload(export: dict[str, Any], domain_label: str) -> dict[str, Any]:
    domain_payload = export["domains"][domain_label]
    return {
        "domain": domain_label,
        "cell_name": domain_payload["domain"]["name"],
        "group_constants_si": convert_group_constants_to_si(domain_label, domain_payload["group_constants"]),
        "delayed_neutrons": build_delayed_neutron_payload(export, domain_label),
    }


def build_overlay_zones(export: dict[str, Any], geometry: dict[str, Any]) -> list[OverlayZone]:
    fuel_ring_labels = sorted_resolved_fuel_ring_labels(export)
    if not fuel_ring_labels:
        return [
            OverlayZone(
                label="core",
                region="core",
                r_min_m=0.0,
                r_max_m=float(geometry["core_radius_m"]),
                z_min_m=-0.5 * float(geometry["core_height_m"]),
                z_max_m=0.5 * float(geometry["core_height_m"]),
            ),
            OverlayZone(
                label="moderator",
                region="moderator",
                r_min_m=0.0,
                r_max_m=float(geometry["moderator_radius_m"]),
                z_min_m=-0.5 * float(geometry["moderator_height_m"]),
                z_max_m=0.5 * float(geometry["moderator_height_m"]),
            ),
        ]

    core_height_m = float(geometry["core_height_m"])
    fuel_element_height_m = float(geometry["fuel_element_height_m"])
    moderator_height_m = float(geometry["moderator_height_m"])
    moderator_radius_m = float(geometry["moderator_radius_m"])
    core_radius_m = float(geometry["core_radius_m"])
    central_radius_m = float(geometry["central_channel_radius_m"])

    zones = [
        OverlayZone(
            label="moderator",
            region="moderator",
            r_min_m=0.0,
            r_max_m=moderator_radius_m,
            z_min_m=-0.5 * moderator_height_m,
            z_max_m=0.5 * moderator_height_m,
        ),
        OverlayZone(
            label="core_heavy_water_coolant_and_moderator",
            region="core_heavy_water_coolant_and_moderator",
            r_min_m=0.0,
            r_max_m=core_radius_m,
            z_min_m=-0.5 * fuel_element_height_m,
            z_max_m=0.5 * fuel_element_height_m,
        ),
        OverlayZone(
            label="core_central_moderator_channel",
            region="core_central_moderator_channel",
            r_min_m=0.0,
            r_max_m=central_radius_m,
            z_min_m=-0.5 * fuel_element_height_m,
            z_max_m=0.5 * core_height_m,
        ),
    ]

    ring_geometry_by_label = {
        f"core_{entry['cell_name']}": entry
        for entry in geometry["fuel_rings"]
    }
    for label in fuel_ring_labels:
        bounds = ring_geometry_by_label[label]
        zones.append(
            OverlayZone(
                label=label,
                region=label,
                r_min_m=float(bounds["r_min_m"]),
                r_max_m=float(bounds["r_max_m"]),
                z_min_m=float(bounds["z_min_m"]),
                z_max_m=float(bounds["z_max_m"]),
            )
        )

    if "core_control_rod" in export["domains"]:
        zones.append(
            OverlayZone(
                label="core_control_rod",
                region="core_control_rod",
                r_min_m=0.0,
                r_max_m=float(geometry["rod_radius_m"]),
                z_min_m=float(geometry["parked_rod_z_min_m"]),
                z_max_m=float(geometry["parked_rod_z_max_m"]),
            )
        )

    return zones


def _unique_sorted(values: list[float], digits: int = 10) -> list[float]:
    deduplicated = sorted({round(value, digits) for value in values})
    return [float(value) for value in deduplicated]


def build_mesh_blocks(geometry: dict[str, Any], overlay_zones: list[OverlayZone]) -> tuple[list[float], list[float], list[MeshBlock]]:
    reflector_radius_m = float(geometry["reflector_radius_m"])
    reflector_height_m = float(geometry["reflector_height_m"])
    radial_edges = [0.0, reflector_radius_m]
    axial_edges = [-0.5 * reflector_height_m, 0.5 * reflector_height_m]
    for zone in overlay_zones:
        radial_edges.extend([zone.r_min_m, zone.r_max_m])
        axial_edges.extend([zone.z_min_m, zone.z_max_m])

    radial_edges = _unique_sorted(radial_edges)
    axial_edges = _unique_sorted(axial_edges)

    blocks: list[MeshBlock] = []
    for radial_index, (r_min_m, r_max_m) in enumerate(zip(radial_edges[:-1], radial_edges[1:], strict=True), start=1):
        if r_max_m <= r_min_m:
            continue
        r_mid_m = 0.5 * (r_min_m + r_max_m)
        for axial_index, (z_min_m, z_max_m) in enumerate(zip(axial_edges[:-1], axial_edges[1:], strict=True), start=1):
            if z_max_m <= z_min_m:
                continue
            z_mid_m = 0.5 * (z_min_m + z_max_m)
            region = "reflector"
            for zone in overlay_zones:
                if zone.contains(r_mid_m, z_mid_m):
                    region = zone.region
            blocks.append(
                MeshBlock(
                    block_id=f"r{radial_index:02d}_z{axial_index:02d}",
                    region=region,
                    r_min_m=r_min_m,
                    r_max_m=r_max_m,
                    z_min_m=z_min_m,
                    z_max_m=z_max_m,
                )
            )
    return radial_edges, axial_edges, blocks


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{float(value):.12g}"


def _format_scalar_list(values: list[float]) -> str:
    return " ".join(_format_number(value) for value in values)


def _foam_header(class_name: str, object_name: str, version: str = "2.0") -> str:
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\n"
        "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
        "|  \\\\    /   O peration     | Version:  2312                                  |\n"
        "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
        "|    \\\\/     M anipulation  |                                                 |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        f"    version         {version};\n"
        "    format          ascii;\n"
        f"    class           {class_name};\n"
        f"    object          {object_name};\n"
        "}\n"
        "\n"
    )


def _dictionary_header(object_name: str) -> str:
    return _foam_header("dictionary", object_name)


def _vol_scalar_header(object_name: str) -> str:
    return _foam_header("volScalarField", object_name, version="2")


def _zone_names(case_spec: dict[str, Any]) -> list[str]:
    return list(case_spec["config"]["domain_order"])


def _unique_region_names(case_spec: dict[str, Any]) -> list[str]:
    return [name for name in _zone_names(case_spec) if name in case_spec["materials"]]


def _prompt_generation_time(case_spec: dict[str, Any]) -> float:
    generation_time = case_spec.get("openmc_run", {}).get("generation_time_s")
    if isinstance(generation_time, dict):
        mean_value = generation_time.get("mean")
        if mean_value is not None:
            return float(mean_value)
    return 1.0e-6


def _prompt_chi(material_payload: dict[str, Any]) -> list[float]:
    return [float(value) for value in material_payload["group_constants_si"]["chi"]["mean"]]


def _delayed_chi(material_payload: dict[str, Any], group_count: int) -> list[float]:
    delayed_summary = material_payload["delayed_neutrons"]["summary"]
    raw = [float(value) for value in delayed_summary.get("beta_total_by_energy_group", [])]
    if len(raw) != group_count:
        return _prompt_chi(material_payload)
    total = sum(raw)
    if total > 0.0:
        return [value / total for value in raw]
    return _prompt_chi(material_payload)


def _beta_by_precursor_group(material_payload: dict[str, Any], precursor_group_count: int) -> list[float]:
    values = material_payload["delayed_neutrons"]["summary"].get("beta_total_by_delayed_group", [])
    return [float(value) for value in values[:precursor_group_count]] + [0.0] * max(0, precursor_group_count - len(values))


def _lambda_by_precursor_group(material_payload: dict[str, Any], precursor_group_count: int) -> list[float]:
    values = material_payload["delayed_neutrons"]["summary"].get("decay_rate_per_s_by_delayed_group", [])
    return [float(value) for value in values[:precursor_group_count]] + [0.0] * max(0, precursor_group_count - len(values))


def _reference_lambda_values(case_spec: dict[str, Any], precursor_group_count: int) -> list[float]:
    for zone_name in _unique_region_names(case_spec):
        values = _lambda_by_precursor_group(case_spec["materials"][zone_name], precursor_group_count)
        if any(value > 0.0 for value in values):
            return [value if value > 0.0 else 1.0e-6 for value in values]
    return [1.0e-6] * precursor_group_count


def _sanitize_lambda_values(values: list[float], fallback_values: list[float]) -> list[float]:
    sanitized: list[float] = []
    for index, value in enumerate(values):
        if value > 0.0 and math.isfinite(value):
            sanitized.append(value)
            continue
        fallback = fallback_values[index] if index < len(fallback_values) else 1.0e-6
        sanitized.append(fallback if fallback > 0.0 and math.isfinite(fallback) else 1.0e-6)
    return sanitized


def _sigma_removal(material_payload: dict[str, Any], group_count: int) -> list[float]:
    absorption = material_payload["group_constants_si"]["absorption"]["mean"]
    scatter = material_payload["group_constants_si"]["scatter matrix"]["mean"]
    values: list[float] = []
    for group_index in range(group_count):
        scatter_out = sum(float(scatter[group_index][destination]) for destination in range(group_count) if destination != group_index)
        values.append(float(absorption[group_index]) + scatter_out)
    positive_values = [value for value in values if value > 0.0 and math.isfinite(value)]
    replacement = min(positive_values) if positive_values else 1.0e-6
    return [value if value > 0.0 and math.isfinite(value) else replacement for value in values]


def _sigma_pow(material_payload: dict[str, Any]) -> list[float]:
    return [float(value) for value in material_payload["group_constants_si"]["nu-fission"]["mean"]]


def _identity_disc_factors(group_count: int) -> list[float]:
    return [1.0] * group_count


def _unity_integral_flux(group_count: int) -> list[float]:
    return [1.0] * group_count


def build_control_dict_text() -> str:
    return (
        _dictionary_header("controlDict")
        + "application     GeN-Foam;\n\n"
        + "startFrom       latestTime;\n\n"
        + "stopAt          endTime;\n\n"
        + "endTime         1;\n\n"
        + "deltaT          1e-09;\n\n"
        + "writeControl    adjustableRunTime;\n\n"
        + "writeInterval   1;\n\n"
        + "purgeWrite      0;\n\n"
        + "writeFormat     ascii;\n\n"
        + "writePrecision  8;\n\n"
        + "writeCompression false;\n\n"
        + "timeFormat      general;\n\n"
        + "timePrecision   8;\n\n"
        + "runTimeModifiable true;\n\n"
        + "removeBaffles\n{\n    neutroRegion     false;\n}\n\n"
        + "adjustTimeStep  true;\n\n"
        + "maxDeltaT       1;\n\n"
        + "maxCo           5;\n\n"
        + "maxPowerVariation 0.01;\n\n"
    )


def build_regions_dict_text() -> str:
    return (
        _dictionary_header("regionsDict")
        + "regionSolvers\n{\n    Level_0\n    {\n        neutroRegion     diffusionNeutronics;\n    }\n}\n\n"
        + "mappings\n{\n    neutroRegion\n    {\n    }\n}\n\n"
    )


def build_fv_schemes_text() -> str:
    return (
        _dictionary_header("fvSchemes")
        + "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        + "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        + "divSchemes\n{\n    default         Gauss linear;\n    \"div(facePhi_,angularFlux_)\"   Gauss upwind;\n}\n\n"
        + "laplacianSchemes\n{\n   default         Gauss linear corrected;\n}\n\n"
        + "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        + "snGradSchemes\n{\n    default         corrected;\n}\n\n"
        + "fluxRequired\n{\n    default         false;\n}\n\n"
    )


def build_fv_solution_text() -> str:
    return (
        _dictionary_header("fvSolution")
        + "solvers\n{\n"
        + "    \"prec.*|precStar.*|adjoint_prec.*\"\n"
        + "    {\n        solver           PBiCG;\n        preconditioner   DILU;\n        tolerance        1e-6;\n        relTol           1e-3;\n    }\n"
        + "    \"flux.*|adjoint_flux.*\"\n"
        + "    {\n        solver          PCG;\n        preconditioner  DIC;\n        tolerance       1e-6;\n        relTol          1e-3;\n    }\n"
        + "    \"angularFlux.*\"\n"
        + "    {\n        solver          PBiCGStab;\n        preconditioner  DILU;\n        tolerance       1e-7;\n        relTol          1e-1;\n    }\n"
        + "}\n\n"
        + "neutronTransport\n{\n"
        + "    integralPredictor           false;\n"
        + "    implicitPredictor           false;\n"
        + "    ROMAcceleration             false;\n"
        + "    aitkenAcceleration          false;\n"
        + "    neutronIterationResidual    0.000001;\n"
        + "    maxNeutronIterations        50;\n"
        + "}\n\n"
    )


def build_neutronics_properties_text() -> str:
    return (
        _dictionary_header("neutronicsProperties")
        + "eigenvalueNeutronics        true;\n\n"
        + "externalSourceNeutronics    false;\n\n"
        + "fastNeutrons                true;\n\n"
        + "adjustDiscFactors           false;\n\n"
        + "useGivenDiscFactors         false;\n\n"
    )


def build_default_flux_field_text(object_name: str) -> str:
    return (
        _vol_scalar_header(object_name)
        + "dimensions      [ 0 -2 -1 0 0 0 0 ];\n\n"
        + "internalField   uniform 1;\n\n"
        + "boundaryField\n{\n"
        + "    wedge_front\n    {\n        type            wedge;\n    }\n"
        + "    wedge_back\n    {\n        type            wedge;\n    }\n"
        + "    axis\n    {\n        type            symmetryPlane;\n    }\n"
        + "    bottom\n    {\n        type            fixedValue;\n        value           uniform 0;\n    }\n"
        + "    top\n    {\n        type            fixedValue;\n        value           uniform 0;\n    }\n"
        + "    outer\n    {\n        type            fixedValue;\n        value           uniform 0;\n    }\n"
        + "}\n\n"
    )


def _zone_block_counts(case_spec: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    mesh_info = case_spec["mesh"]
    if "blocks" in mesh_info:
        for block in mesh_info["blocks"]:
            region = block["region"]
            counts[region] = counts.get(region, 0) + 1
        return counts

    for mapping in mesh_info.get("region_to_cell_zone", []):
        region = mapping["region"]
        counts[region] = counts.get(region, 0) + 1
    return counts


def build_nuclear_data_text(case_spec: dict[str, Any]) -> str:
    group_count = int(case_spec["config"]["group_count"])
    precursor_group_count = len(case_spec["config"]["delayed_groups"])
    zone_counts = _zone_block_counts(case_spec)
    reference_lambda_values = _reference_lambda_values(case_spec, precursor_group_count)
    lines = [
        _dictionary_header("nuclearData").rstrip(),
        f"promptGenerationTime {_format_number(_prompt_generation_time(case_spec))};",
        f"precGroups      {precursor_group_count};",
        f"energyGroups    {group_count};",
        "",
        "xsVariables",
        "{}",
        "",
        "states",
        "(",
        "    reference",
        "    {",
        "        zones",
        "        (",
    ]

    for zone_name in _unique_region_names(case_spec):
        material_payload = case_spec["materials"][zone_name]
        group_constants = material_payload["group_constants_si"]
        prompt_chi = _prompt_chi(material_payload)
        delayed_chi = _delayed_chi(material_payload, group_count)
        removal = _sigma_removal(material_payload, group_count)
        nu_sigma_eff = [float(value) for value in group_constants["nu-fission"]["mean"]]
        sigma_pow = _sigma_pow(material_payload)
        scatter = group_constants["scatter matrix"]["mean"]
        diffusion = [float(value) for value in group_constants["diffusion-coefficient"]["mean"]]
        inverse_velocity = [float(value) for value in group_constants["inverse-velocity"]["mean"]]
        lambda_values = _sanitize_lambda_values(
            _lambda_by_precursor_group(material_payload, precursor_group_count),
            reference_lambda_values,
        )
        beta_values = _beta_by_precursor_group(material_payload, precursor_group_count)
        zone_block_count = zone_counts.get(zone_name, 0)

        lines.extend([
            f"            {zone_name}",
            "            {",
            f"                // Populated from {zone_block_count} OpenMC-derived 3D cell zone mapping(s).",
            "                fuelFraction    1;",
            f"                sigmaRemoval    nonuniform List<scalar> {group_count} ({_format_scalar_list(removal)});",
            f"                nuSigmaEff      nonuniform List<scalar> {group_count} ({_format_scalar_list(nu_sigma_eff)});",
            f"                sigmaPow        nonuniform List<scalar> {group_count} ({_format_scalar_list(sigma_pow)});",
            f"                scatteringMatrixP0 {group_count} {group_count}",
            "                (",
        ])
        for row in scatter:
            lines.append(f"                    ( {_format_scalar_list([float(value) for value in row])} )")
        lines.extend([
            "                );",
            f"                discFactor      nonuniform List<scalar> {group_count} ({_format_scalar_list(_identity_disc_factors(group_count))});",
            f"                chiPrompt       nonuniform List<scalar> {group_count} ({_format_scalar_list(prompt_chi)});",
            f"                chiDelayed      nonuniform List<scalar> {group_count} ({_format_scalar_list(delayed_chi)});",
            f"                IV              nonuniform List<scalar> {group_count} ({_format_scalar_list(inverse_velocity)});",
            f"                D               nonuniform List<scalar> {group_count} ({_format_scalar_list(diffusion)});",
            f"                integralFlux    nonuniform List<scalar> {group_count} ({_format_scalar_list(_unity_integral_flux(group_count))});",
            f"                lambda          nonuniform List<scalar> {precursor_group_count} ({_format_scalar_list(lambda_values)});",
            f"                Beta            nonuniform List<scalar> {precursor_group_count} ({_format_scalar_list(beta_values)});",
            "            }",
        ])

    lines.extend([
        "        );",
        "    }",
        ");",
        "",
    ])
    return "\n".join(lines) + "\n"
def build_allclean_text() -> str:
    return (
        "#!/bin/sh\n"
        "cd ${0%/*} || exit 1\n\n"
        "rm -rf 0.0 [1-9]* constant/neutroRegion/polyMesh\n"
        "rm -rf log.* *.log *.out\n"
    )


def build_allmesh_text() -> str:
    mesh_file = Path("constant/generated/concentric_reactor_wedge.msh")
    mesh_manifest = Path("constant/generated/concentric_reactor_mesh_manifest.json")
    return (
        "#!/bin/sh\n"
        "cd ${0%/*} || exit 1\n\n"
        ". $WM_PROJECT_DIR/bin/tools/RunFunctions\n\n"
        "rm -rf log.gmshGenerate log.gmshToFoam log.configureMesh log.checkMesh log.validateMesh\n"
        "rm -rf constant/neutroRegion/polyMesh\n"
        f"runApplication -s log.gmshGenerate conda run -n openmc python gmsh_reactor_mesh.py generate --mesh-kind {WEDGE_MESH_KIND} --mesh-file {mesh_file} --manifest-file {mesh_manifest}\n"
        f"runApplication -s log.gmshToFoam gmshToFoam -case . -region neutroRegion {mesh_file}\n"
        f"runApplication -s log.configureMesh conda run -n openmc python gmsh_reactor_mesh.py configure-import --case-dir . --region neutroRegion --manifest-file {mesh_manifest}\n"
        "runApplication checkMesh -case . -region neutroRegion\n"
        f"runApplication -s log.validateMesh conda run -n openmc python gmsh_reactor_mesh.py validate-import --case-dir . --region neutroRegion --manifest-file {mesh_manifest}\n"
    )


def build_allrun_text() -> str:
    return (
        "#!/bin/sh\n"
        "cd ${0%/*} || exit 1\n\n"
        ". $WM_PROJECT_DIR/bin/tools/RunFunctions\n\n"
        "./Allmesh || exit 1\n"
        "runApplication GeN-Foam -case .\n"
    )


def write_openfoam_case(case_spec: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    group_count = int(case_spec["config"]["group_count"])
    output_paths = {
        "controlDict": case_dir / "system" / "controlDict",
        "regionsDict": case_dir / "system" / "regionsDict",
        "fvSchemes": case_dir / "system" / "neutroRegion" / "fvSchemes",
        "fvSolution": case_dir / "system" / "neutroRegion" / "fvSolution",
        "neutronicsProperties": case_dir / "constant" / "neutroRegion" / "neutronicsProperties",
        "nuclearData": case_dir / "constant" / "neutroRegion" / "nuclearData",
        "Allclean": case_dir / "Allclean",
        "Allmesh": case_dir / "Allmesh",
        "Allrun": case_dir / "Allrun",
    }

    write_text(output_paths["controlDict"], build_control_dict_text())
    write_text(output_paths["regionsDict"], build_regions_dict_text())
    write_text(output_paths["fvSchemes"], build_fv_schemes_text())
    write_text(output_paths["fvSolution"], build_fv_solution_text())
    write_text(output_paths["neutronicsProperties"], build_neutronics_properties_text())
    write_text(output_paths["nuclearData"], build_nuclear_data_text(case_spec))
    write_text(output_paths["Allclean"], build_allclean_text())
    write_text(output_paths["Allmesh"], build_allmesh_text())
    write_text(output_paths["Allrun"], build_allrun_text())
    for script_name in ("Allclean", "Allmesh", "Allrun"):
        output_paths[script_name].chmod(0o755)

    legacy_block_mesh = case_dir / "system" / "neutroRegion" / "blockMeshDict"
    if legacy_block_mesh.exists():
        legacy_block_mesh.unlink()

    uniform_dir = case_dir / "0" / "uniform"
    uniform_dir.mkdir(parents=True, exist_ok=True)
    for child in uniform_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    default_flux_paths: list[str] = []
    zero_dir = case_dir / "0" / "neutroRegion"
    zero_dir.mkdir(parents=True, exist_ok=True)
    existing_default_flux = [path for path in zero_dir.iterdir() if path.name.startswith("defaultFlux")]
    for path in existing_default_flux:
        if path.is_file():
            path.unlink()
    external_source_path = zero_dir / "defaultExternalSourceFlux"
    if external_source_path.exists():
        external_source_path.unlink()

    for group_index in range(group_count):
        object_name = "defaultFlux" if group_index == 0 else f"defaultFlux{group_index + 1}"
        flux_path = zero_dir / object_name
        write_text(flux_path, build_default_flux_field_text(object_name))
        default_flux_paths.append(str(flux_path))

    output_paths["defaultFlux"] = zero_dir / "defaultFlux"
    return output_paths | {"fluxFiles": default_flux_paths}


def build_case_spec(mgxs_export_dir: Path, output_dir: Path) -> dict[str, Any]:
    mgxs_export_dir = require_dir(mgxs_export_dir)
    model_xml_path = require_file(mgxs_export_dir / "reactor_run" / "model.xml")
    mgxs_json_path = require_file(mgxs_export_dir / "outputs" / "mgxs_constants.json")
    export = load_json(mgxs_json_path)
    manual_geometry = default_reactor_geometry()
    geometry = manual_geometry_payload(manual_geometry)
    region_mapping = mesh_region_mapping(manual_geometry)
    mesh_info = gmsh_mesh_metadata(
        mesh_file=output_dir / "concentric_reactor_wedge.msh",
        manifest_file=output_dir / "concentric_reactor_mesh_manifest.json",
        geometry=manual_geometry,
        mesh_kind=WEDGE_MESH_KIND,
    )

    domain_order = [label for label, _ in export["config"]["domain_definitions"]]
    configured_regions = {item["region"] for item in region_mapping}
    missing_regions = [label for label in domain_order if label not in configured_regions]
    if missing_regions:
        raise ValueError(f"Manual Gmsh geometry is missing configured regions for: {', '.join(missing_regions)}")
    materials = {
        domain_label: build_material_payload(export, domain_label)
        for domain_label in domain_order
    }

    return {
        "source": {
            "mgxs_export_dir": str(mgxs_export_dir),
            "model_xml_path": str(model_xml_path),
            "mgxs_constants_json": str(mgxs_json_path),
        },
        "config": {
            "group_count": len(export["config"]["energy_group_edges_ev"]) - 1,
            "energy_group_edges_ev": export["config"]["energy_group_edges_ev"],
            "legendre_order": export["config"]["legendre_order"],
            "domain_order": domain_order,
            "delayed_groups": export["config"]["delayed_groups"],
        },
        "geometry": geometry,
        "mesh_regions": region_mapping,
        "mesh": mesh_info,
        "materials": materials,
        "openmc_run": export.get("run", {}),
    }


def write_case_outputs(case_spec: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": output_dir / "concentric_case_manifest.json",
        "mesh_manifest": output_dir / "concentric_reactor_mesh_manifest.json",
        "mesh_file": output_dir / "concentric_reactor_wedge.msh",
        "zones_csv": output_dir / "concentric_mesh_regions.csv",
        "materials": output_dir / "concentric_materials.json",
    }
    write_mesh_assets(
        mesh_file=files["mesh_file"],
        manifest_file=files["mesh_manifest"],
        mesh_kind=WEDGE_MESH_KIND,
    )
    write_json(files["manifest"], case_spec)
    write_json(files["materials"], {"materials": case_spec["materials"], "config": case_spec["config"]})
    write_csv(
        files["zones_csv"],
        rows=case_spec["mesh_regions"],
        fieldnames=["region", "cell_zone"],
    )
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a GeN-Foam-ready concentric reactor case from OpenMC MGXS exports."
    )
    parser.add_argument(
        "--mgxs-export-dir",
        type=Path,
        default=DEFAULT_MGXS_EXPORT_DIR,
        help="Path to an OpenMC MGXS export directory containing reactor_run/model.xml and outputs/mgxs_constants.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated case metadata should be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_spec = build_case_spec(args.mgxs_export_dir, args.output_dir)
    files = write_case_outputs(case_spec, args.output_dir)
    openfoam_files = write_openfoam_case(case_spec, CASE_DIR)

    rendered_openfoam_files: dict[str, Any] = {}
    for name, value in openfoam_files.items():
        if isinstance(value, list):
            rendered_openfoam_files[name] = value
        else:
            rendered_openfoam_files[name] = str(value)

    print(
        json.dumps(
            {
                "group_count": case_spec["config"]["group_count"],
                "mesh_region_count": len(case_spec["mesh_regions"]),
                "output_files": {name: str(path) for name, path in files.items()},
                "openfoam_case_files": rendered_openfoam_files,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
