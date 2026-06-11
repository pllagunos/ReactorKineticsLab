"""Load resolved OpenMC MGXS exports into the multigroup diffusion solver."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .multigroup_diffusion import (
    CylindricalLayeredModel2D,
    CylindricalRegionZone2D,
    MultiGroupRegion,
)


_FUEL_RING_LABEL = re.compile(r"^core_fuel_ring_(\d+)$")
_FUEL_RING_CELL = re.compile(r"^fuel_ring_(\d+)$")
_SCATTER_MATRIX_PRIORITY = ("consistent scatter matrix", "scatter matrix")
_REQUIRED_DOMAIN_LABELS = {
    "core_central_moderator_channel",
    "core_heavy_water_coolant_and_moderator",
    "moderator",
    "reflector",
}
_MAX_DIFFUSION_CM = 1.0e6
_DIFFUSION_REL_TOL = 1.0e-8
_DIFFUSION_ABS_TOL = 1.0e-12


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find required file: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _parse_surface_value(surface: ET.Element) -> float:
    coefficients = [float(token) for token in surface.attrib["coeffs"].split()]
    surface_type = surface.attrib["type"]
    if surface_type == "z-cylinder":
        return coefficients[-1]
    if surface_type == "z-plane":
        return coefficients[0]
    raise ValueError(f"Unsupported surface type {surface_type!r}")


def _named_cells(root: ET.Element) -> dict[str, ET.Element]:
    cells: dict[str, ET.Element] = {}
    for cell in root.findall(".//cell"):
        name = cell.attrib.get("name")
        if not name:
            continue
        if name in cells:
            raise ValueError(f"model.xml contains duplicate cell name {name!r}")
        cells[name] = cell
    return cells


def _extract_cell_bounds(
    cell_name: str,
    cells: dict[str, ET.Element],
    surfaces: dict[str, ET.Element],
) -> dict[str, float]:
    try:
        cell = cells[cell_name]
    except KeyError as exc:
        raise ValueError(f"Could not find cell {cell_name!r} in model.xml") from exc

    surface_ids = {
        token.lstrip("-")
        for token in re.findall(r"-?\d+", cell.attrib.get("region", ""))
    }
    try:
        cell_surfaces = [surfaces[surface_id] for surface_id in surface_ids]
    except KeyError as exc:
        raise ValueError(
            f"Cell {cell_name!r} references missing surface {exc.args[0]!r}"
        ) from exc

    radii = sorted(
        _parse_surface_value(surface)
        for surface in cell_surfaces
        if surface.attrib["type"] == "z-cylinder"
    )
    z_planes = sorted(
        _parse_surface_value(surface)
        for surface in cell_surfaces
        if surface.attrib["type"] == "z-plane"
    )
    if not z_planes:
        raise ValueError(f"Cell {cell_name!r} does not define axial bounds")

    return {
        "r_min_cm": radii[0] if len(radii) > 1 else 0.0,
        "r_max_cm": radii[-1] if radii else 0.0,
        "z_min_cm": z_planes[0],
        "z_max_cm": z_planes[-1],
        "height_cm": z_planes[-1] - z_planes[0],
    }


def _sorted_numbered_names(
    names: list[str],
    pattern: re.Pattern[str],
) -> list[str]:
    numbered: list[tuple[int, str]] = []
    for name in names:
        match = pattern.match(name)
        if match is not None:
            numbered.append((int(match.group(1)), name))
    return [name for _, name in sorted(numbered)]


def _extract_geometry(model_xml_path: Path) -> tuple[dict[str, Any], set[str]]:
    root = ET.parse(model_xml_path).getroot()
    cells = _named_cells(root)
    surfaces = {
        surface.attrib["id"]: surface
        for surface in root.findall(".//surface")
    }

    fuel_element = _extract_cell_bounds("fuel_element", cells, surfaces)
    moderator = _extract_cell_bounds("d2o_tank", cells, surfaces)
    reflector = _extract_cell_bounds("h2o_tank", cells, surfaces)
    control_rod = _extract_cell_bounds("control_rod", cells, surfaces)
    central_channel = _extract_cell_bounds(
        "central_moderator_channel", cells, surfaces
    )

    fuel_ring_names = _sorted_numbered_names(list(cells), _FUEL_RING_CELL)
    if not fuel_ring_names:
        raise ValueError(
            "Resolved diffusion input requires cells named fuel_ring_<index>"
        )
    fuel_rings = [
        _extract_cell_bounds(name, cells, surfaces)
        for name in fuel_ring_names
    ]
    first_ring = fuel_rings[0]
    if any(
        not math.isclose(bounds["height_cm"], first_ring["height_cm"])
        for bounds in fuel_rings[1:]
    ):
        raise ValueError("All fuel rings must use the same active height")

    geometry: dict[str, Any] = {
        "fuel_element_radius_cm": fuel_element["r_max_cm"],
        "fuel_element_height_cm": fuel_element["height_cm"],
        "core_radius_cm": fuel_element["r_max_cm"],
        "core_height_cm": first_ring["height_cm"],
        "axial_moderator_extension_cm": 0.5
        * (fuel_element["height_cm"] - first_ring["height_cm"]),
        "central_channel_radius_cm": central_channel["r_max_cm"],
        "moderator_radius_cm": moderator["r_max_cm"],
        "moderator_height_cm": moderator["height_cm"],
        "reflector_radius_cm": reflector["r_max_cm"],
        "reflector_height_cm": reflector["height_cm"],
        "outer_height_cm": max(
            moderator["height_cm"], reflector["height_cm"]
        ),
        "rod_radius_cm": control_rod["r_max_cm"],
        "parked_rod_z_min_cm": control_rod["z_min_cm"],
        "parked_rod_z_max_cm": control_rod["z_max_cm"],
        "fuel_rings": [
            {
                "cell_name": name,
                "r_min_cm": bounds["r_min_cm"],
                "r_max_cm": bounds["r_max_cm"],
                "z_min_cm": bounds["z_min_cm"],
                "z_max_cm": bounds["z_max_cm"],
            }
            for name, bounds in zip(fuel_ring_names, fuel_rings, strict=True)
        ],
    }
    return geometry, set(cells)


def _domain_mapping(export: dict[str, Any]) -> dict[str, str]:
    definitions = export.get("config", {}).get("domain_definitions")
    if not isinstance(definitions, list):
        raise ValueError("MGXS export is missing config.domain_definitions")

    mapping: dict[str, str] = {}
    for definition in definitions:
        if not isinstance(definition, list) or len(definition) != 2:
            raise ValueError(f"Invalid domain definition: {definition!r}")
        label, cell_name = (str(item) for item in definition)
        if label in mapping:
            raise ValueError(f"Duplicate MGXS domain label {label!r}")
        mapping[label] = cell_name
    return mapping


def _validate_export_contract(
    export: dict[str, Any],
    xml_cell_names: set[str],
) -> tuple[dict[str, str], list[str]]:
    config = export.get("config", {})
    if "scatter_correction" not in config:
        raise ValueError(
            "MGXS export does not declare config.scatter_correction. "
            "Regenerate it with the current openmc/mgxs_export.py."
        )
    if config["scatter_correction"] is not None:
        raise ValueError(
            "Diffusion input requires scatter_correction = null so the P0 "
            "scattering diagonal is not transport-corrected."
        )

    domains = export.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError("MGXS export does not contain any domains")

    mapping = _domain_mapping(export)
    if set(mapping) != set(domains):
        raise ValueError(
            "config.domain_definitions and exported domains must contain "
            f"the same labels; definitions-only={sorted(set(mapping) - set(domains))}, "
            f"domains-only={sorted(set(domains) - set(mapping))}"
        )
    missing_labels = sorted(_REQUIRED_DOMAIN_LABELS - domains.keys())
    fuel_ring_labels = _sorted_numbered_names(list(domains), _FUEL_RING_LABEL)
    if missing_labels or not fuel_ring_labels:
        details = ", ".join(missing_labels) if missing_labels else "fuel rings"
        raise ValueError(
            "Resolved diffusion input is missing required MGXS domains: "
            f"{details}"
        )

    expected_ring_cells = [f"fuel_ring_{index + 1}" for index in range(len(fuel_ring_labels))]
    mapped_ring_cells = [mapping.get(label) for label in fuel_ring_labels]
    if mapped_ring_cells != expected_ring_cells:
        raise ValueError(
            "Resolved fuel-ring domains do not map in order to "
            f"{expected_ring_cells}; got {mapped_ring_cells}"
        )

    for label, domain_payload in domains.items():
        if label not in mapping:
            raise ValueError(f"Domain {label!r} is missing from domain_definitions")
        cell_name = mapping[label]
        if cell_name not in xml_cell_names:
            raise ValueError(
                f"MGXS domain {label!r} references missing XML cell {cell_name!r}"
            )
        payload_name = domain_payload.get("domain", {}).get("name")
        if payload_name != cell_name:
            raise ValueError(
                f"MGXS domain {label!r} names cell {payload_name!r}, "
                f"but domain_definitions maps it to {cell_name!r}"
            )

    if "control_rod" in xml_cell_names and "core_control_rod" not in domains:
        raise ValueError(
            "Resolved diffusion input is missing core_control_rod MGXS data"
        )
    return mapping, fuel_ring_labels


def _select_scatter_matrix(
    group_constants: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    for xs_type in _SCATTER_MATRIX_PRIORITY:
        payload = group_constants.get(xs_type)
        if isinstance(payload, dict):
            return xs_type, payload
    raise ValueError(
        "MGXS domain is missing a non-nu scattering matrix. Expected "
        "'consistent scatter matrix' or 'scatter matrix'."
    )


def _collapse_p0(values: Any, group_count: int) -> np.ndarray:
    scatter = np.asarray(values, dtype=float)
    if scatter.shape == (group_count, group_count):
        return scatter
    if scatter.ndim == 3 and scatter.shape[1:] == (group_count, group_count):
        return scatter[0]
    if scatter.ndim == 3 and scatter.shape[:2] == (group_count, group_count):
        return scatter[:, :, 0]
    raise ValueError(
        "Scattering matrix must have shape "
        f"{(group_count, group_count)}, (L, {group_count}, {group_count}), "
        f"or ({group_count}, {group_count}, L); got {scatter.shape}"
    )


def _vector(
    group_constants: dict[str, Any],
    xs_type: str,
    group_count: int,
) -> np.ndarray:
    try:
        values = group_constants[xs_type]["mean"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"MGXS domain is missing {xs_type!r}") from exc
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape != (group_count,):
        raise ValueError(
            f"{xs_type!r} must contain {group_count} values, got {vector.shape}"
        )
    return vector


def _inactive_group(
    group: int,
    absorption: np.ndarray,
    nu_fission: np.ndarray,
    chi: np.ndarray,
    scatter: np.ndarray,
) -> bool:
    return (
        absorption[group] == 0.0
        and nu_fission[group] == 0.0
        and chi[group] == 0.0
        and np.all(scatter[group, :] == 0.0)
        and np.all(scatter[:, group] == 0.0)
    )


def _region_from_domain(
    label: str,
    domain_payload: dict[str, Any],
    group_count: int,
) -> tuple[MultiGroupRegion, dict[str, Any]]:
    group_constants = domain_payload.get("group_constants")
    if not isinstance(group_constants, dict):
        raise ValueError(f"MGXS domain {label!r} has no group_constants")

    scatter_source, scatter_payload = _select_scatter_matrix(group_constants)
    scatter = _collapse_p0(scatter_payload.get("mean"), group_count)
    absorption = _vector(group_constants, "absorption", group_count)
    nu_fission = _vector(group_constants, "nu-fission", group_count)
    chi = _vector(group_constants, "chi", group_count)
    diffusion = _vector(group_constants, "diffusion-coefficient", group_count)
    transport = _vector(group_constants, "transport", group_count)

    for field_name, values in (
        ("absorption", absorption),
        ("nu-fission", nu_fission),
        ("chi", chi),
        (scatter_source, scatter),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"MGXS domain {label!r} contains non-finite {field_name} values"
            )

    replacements: list[dict[str, float | int]] = []
    checked_relative_errors: list[float] = []
    for group in range(group_count):
        inactive = _inactive_group(
            group, absorption, nu_fission, chi, scatter
        )
        valid_diffusion = (
            math.isfinite(diffusion[group])
            and 0.0 < diffusion[group] <= _MAX_DIFFUSION_CM
        )
        valid_transport = math.isfinite(transport[group]) and transport[group] > 0.0

        if inactive and not valid_diffusion:
            replacement = next(
                (
                    float(candidate)
                    for other_group, candidate in enumerate(diffusion)
                    if other_group != group
                    and math.isfinite(candidate)
                    and 0.0 < candidate <= _MAX_DIFFUSION_CM
                ),
                1.0,
            )
            replacements.append(
                {
                    "group": group + 1,
                    "original": float(diffusion[group]),
                    "replacement": replacement,
                }
            )
            diffusion[group] = replacement
            continue

        if not valid_diffusion:
            raise ValueError(
                f"MGXS domain {label!r} has invalid diffusion coefficient "
                f"in active group {group + 1}"
            )
        if not valid_transport:
            if inactive:
                continue
            raise ValueError(
                f"MGXS domain {label!r} has invalid transport cross section "
                f"in active group {group + 1}"
            )

        expected = 1.0 / (3.0 * transport[group])
        relative_error = abs(diffusion[group] - expected) / expected
        if not math.isclose(
            diffusion[group],
            expected,
            rel_tol=_DIFFUSION_REL_TOL,
            abs_tol=_DIFFUSION_ABS_TOL,
        ):
            raise ValueError(
                f"MGXS domain {label!r} group {group + 1} has "
                f"D={diffusion[group]:.12g}, but 1/(3*Sigma_tr)="
                f"{expected:.12g}"
            )
        checked_relative_errors.append(relative_error)

    region = MultiGroupRegion(
        name=label,
        diffusion=diffusion,
        absorption=absorption,
        nu_fission=nu_fission,
        chi=chi,
        scatter=scatter,
    )
    return region, {
        "scatter_matrix_source": scatter_source,
        "diffusion_checked_groups": len(checked_relative_errors),
        "diffusion_max_relative_error": (
            max(checked_relative_errors) if checked_relative_errors else 0.0
        ),
        "diffusion_replacements": replacements,
    }


def _build_zones(
    geometry: dict[str, Any],
    regions: dict[str, MultiGroupRegion],
    fuel_ring_labels: list[str],
    domain_mapping: dict[str, str],
) -> tuple[tuple[CylindricalRegionZone2D, ...], tuple[dict[str, Any], ...]]:
    zones: list[CylindricalRegionZone2D] = []
    report: list[dict[str, Any]] = []

    def add_zone(
        label: str,
        *,
        r_max: float,
        z_min: float,
        z_max: float,
        r_min: float = 0.0,
    ) -> None:
        zones.append(
            CylindricalRegionZone2D(
                region=regions[label],
                r_min=r_min,
                r_max=r_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
        report.append(
            {
                "label": label,
                "cell_name": domain_mapping[label],
                "r_min_cm": r_min,
                "r_max_cm": r_max,
                "z_min_cm": z_min,
                "z_max_cm": z_max,
            }
        )

    add_zone(
        "reflector",
        r_max=geometry["reflector_radius_cm"],
        z_min=-0.5 * geometry["reflector_height_cm"],
        z_max=0.5 * geometry["reflector_height_cm"],
    )
    add_zone(
        "moderator",
        r_max=geometry["moderator_radius_cm"],
        z_min=-0.5 * geometry["moderator_height_cm"],
        z_max=0.5 * geometry["moderator_height_cm"],
    )
    add_zone(
        "core_heavy_water_coolant_and_moderator",
        r_max=geometry["core_radius_cm"],
        z_min=-0.5 * geometry["fuel_element_height_cm"],
        z_max=0.5 * geometry["fuel_element_height_cm"],
    )
    add_zone(
        "core_central_moderator_channel",
        r_max=geometry["central_channel_radius_cm"],
        z_min=-0.5 * geometry["fuel_element_height_cm"],
        z_max=0.5 * geometry["core_height_cm"],
    )

    ring_geometry = {
        f"core_{ring['cell_name']}": ring
        for ring in geometry["fuel_rings"]
    }
    if set(ring_geometry) != set(fuel_ring_labels):
        raise ValueError(
            "Fuel-ring labels in MGXS and model.xml do not match: "
            f"MGXS={fuel_ring_labels}, XML={sorted(ring_geometry)}"
        )
    for label in fuel_ring_labels:
        ring = ring_geometry[label]
        add_zone(
            label,
            r_min=ring["r_min_cm"],
            r_max=ring["r_max_cm"],
            z_min=ring["z_min_cm"],
            z_max=ring["z_max_cm"],
        )

    if "core_control_rod" in regions:
        add_zone(
            "core_control_rod",
            r_max=geometry["rod_radius_cm"],
            z_min=geometry["parked_rod_z_min_cm"],
            z_max=geometry["parked_rod_z_max_cm"],
        )
    return tuple(zones), tuple(report)


@dataclass(frozen=True)
class ConcentricDiffusionInput:
    export_dir: Path
    model_xml_path: Path
    mgxs_json_path: Path
    energy_group_edges_ev: tuple[float, ...]
    source_legendre_order: int
    scatter_correction: str | None
    validation_scatter_correction: str | None
    geometry: dict[str, Any]
    regions: dict[str, MultiGroupRegion]
    zones: tuple[CylindricalRegionZone2D, ...]
    zone_report: tuple[dict[str, Any], ...]
    fuel_ring_labels: tuple[str, ...]
    domain_mapping: dict[str, str]
    validation: dict[str, Any]
    openmc_reference: dict[str, float]

    @property
    def group_count(self) -> int:
        return len(self.energy_group_edges_ev) - 1

    def build_model(
        self,
        delta_absorption_rod: float | np.ndarray = 0.0,
    ) -> CylindricalLayeredModel2D:
        core_reference = self.regions[self.fuel_ring_labels[0]]
        return CylindricalLayeredModel2D(
            core_radius=self.geometry["core_radius_cm"],
            moderator_radius=self.geometry["moderator_radius_cm"],
            reflector_radius=self.geometry["reflector_radius_cm"],
            core_height=self.geometry["core_height_cm"],
            outer_height=self.geometry["outer_height_cm"],
            core=core_reference,
            moderator=self.regions["moderator"],
            reflector=self.regions["reflector"],
            rod_radius=self.geometry["rod_radius_cm"],
            delta_absorption_rod=delta_absorption_rod,
            zones=self.zones,
        )

    def summary(self) -> dict[str, Any]:
        return _json_safe(
            {
                "export_dir": self.export_dir,
                "model_xml_path": self.model_xml_path,
                "mgxs_json_path": self.mgxs_json_path,
                "group_count": self.group_count,
                "energy_group_edges_ev": self.energy_group_edges_ev,
                "source_legendre_order": self.source_legendre_order,
                "diffusion_scattering_moment": "P0",
                "discarded_higher_moments": self.source_legendre_order > 0,
                "scatter_correction": self.scatter_correction,
                "validation_scatter_correction": self.validation_scatter_correction,
                "geometry": self.geometry,
                "domain_mapping": self.domain_mapping,
                "zone_count": len(self.zones),
                "zones": self.zone_report,
                "validation": self.validation,
                "openmc_reference": self.openmc_reference,
            }
        )


def load_concentric_diffusion_input(
    export_dir: str | Path,
) -> ConcentricDiffusionInput:
    export_dir = Path(export_dir).expanduser().resolve()
    if not export_dir.is_dir():
        raise FileNotFoundError(f"Could not find MGXS export directory: {export_dir}")

    model_xml_path = _require_file(export_dir / "reactor_run" / "model.xml")
    mgxs_json_path = _require_file(
        export_dir / "outputs" / "mgxs_constants.json"
    )
    export = _load_json(mgxs_json_path)
    geometry, xml_cell_names = _extract_geometry(model_xml_path)
    mapping, fuel_ring_labels = _validate_export_contract(
        export, xml_cell_names
    )

    config = export["config"]
    edges = tuple(float(value) for value in config["energy_group_edges_ev"])
    if len(edges) < 2 or any(
        upper <= lower for lower, upper in zip(edges, edges[1:])
    ):
        raise ValueError("energy_group_edges_ev must be strictly increasing")
    group_count = len(edges) - 1
    source_legendre_order = int(config.get("legendre_order", 0))

    regions: dict[str, MultiGroupRegion] = {}
    domain_validation: dict[str, Any] = {}
    for label, domain_payload in export["domains"].items():
        region, validation = _region_from_domain(
            label, domain_payload, group_count
        )
        regions[label] = region
        domain_validation[label] = validation

    zones, zone_report = _build_zones(
        geometry, regions, fuel_ring_labels, mapping
    )
    zoned_labels = {zone.region.name for zone in zones}
    if zoned_labels != set(regions):
        raise ValueError(
            "Every exported MGXS region must have one resolved diffusion zone; "
            f"unplaced={sorted(set(regions) - zoned_labels)}, "
            f"unknown_zones={sorted(zoned_labels - set(regions))}"
        )
    run = export.get("run", {})
    keff = run.get("keff", {})
    openmc_reference = {
        "keff": float(keff["mean"]),
        "keff_std_dev": float(keff.get("std_dev", 0.0)),
        "reactivity_pcm": float(run["reactivity_pcm"]),
    }
    scatter_sources = sorted(
        {
            validation["scatter_matrix_source"]
            for validation in domain_validation.values()
        }
    )
    validation = {
        "scatter_matrix_sources": scatter_sources,
        "domains": domain_validation,
        "sanitized_domain_count": sum(
            bool(item["diffusion_replacements"])
            for item in domain_validation.values()
        ),
        "max_diffusion_relative_error": max(
            item["diffusion_max_relative_error"]
            for item in domain_validation.values()
        ),
    }
    return ConcentricDiffusionInput(
        export_dir=export_dir,
        model_xml_path=model_xml_path,
        mgxs_json_path=mgxs_json_path,
        energy_group_edges_ev=edges,
        source_legendre_order=source_legendre_order,
        scatter_correction=config["scatter_correction"],
        validation_scatter_correction=config.get(
            "validation_scatter_correction"
        ),
        geometry=geometry,
        regions=regions,
        zones=zones,
        zone_report=zone_report,
        fuel_ring_labels=tuple(fuel_ring_labels),
        domain_mapping=mapping,
        validation=validation,
        openmc_reference=openmc_reference,
    )
