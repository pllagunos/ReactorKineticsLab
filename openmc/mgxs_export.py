from __future__ import annotations

import copy
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openmc
import openmc.mgxs as mgxs

from ploting import resolve_openmc_exec


DEFAULT_MGXS_TYPES = (
    "total",
    "transport",
    "absorption",
    "diffusion-coefficient",
    "nu-fission",
    "kappa-fission",
    "chi",
    "scatter matrix",
    "nu-scatter matrix",
    "inverse-velocity",
)

DEFAULT_DOMAIN_DEFINITIONS = (
    ("core", "fuel_element"),
    ("moderator", "d2o_tank"),
    ("reflector", "h2o_tank"),
)

SUPPORTED_CORE_MODELING = ("supercell", "resolved")
_FUEL_RING_NAME_PATTERN = re.compile(r"^fuel_ring_(\d+)$")
SUPPORTED_LEGENDRE_ORDERS = (0, 1, 3)
SUPPORTED_SCATTERING_MOMENTS = tuple(f"P{order}" for order in SUPPORTED_LEGENDRE_ORDERS)

DEFAULT_ENERGY_GROUP_EDGES_EV = (0.0, 20.0e6)
TWO_GROUP_ENERGY_GROUP_EDGES_EV = tuple(float(edge) for edge in mgxs.GROUP_STRUCTURES["CASMO-2"])
FOUR_GROUP_ENERGY_GROUP_EDGES_EV = tuple(float(edge) for edge in mgxs.GROUP_STRUCTURES["CASMO-4"])
EIGHT_GROUP_ENERGY_GROUP_EDGES_EV = tuple(float(edge) for edge in mgxs.GROUP_STRUCTURES["CASMO-8"])
CASMO_16_ENERGY_GROUP_EDGES_EV = tuple(float(edge) for edge in mgxs.GROUP_STRUCTURES["CASMO-16"])
CASMO_25_ENERGY_GROUP_EDGES_EV = tuple(float(edge) for edge in mgxs.GROUP_STRUCTURES["CASMO-25"])
CASMO_40_ENERGY_GROUP_EDGES_EV = tuple(float(edge) for edge in mgxs.GROUP_STRUCTURES["CASMO-40"])
CASMO_70_ENERGY_GROUP_EDGES_EV = tuple(float(edge) for edge in mgxs.GROUP_STRUCTURES["CASMO-70"])
SUPPORTED_GROUP_COUNTS = (1, 2, 4, 8, 16, 25, 40, 70)
DEFAULT_DELAYED_GROUPS = (1, 2, 3, 4, 5, 6)
DEFAULT_MODEL_XML_PATH = Path("build") / "concentric" / "model.xml"
DEFAULT_EXPORT_DIRNAME = "mgxs_export"


@dataclass(frozen=True)
class MGXSExportConfig:
    particles: int = 16000
    batches: int = 20
    inactive: int = 5
    domain_definitions: tuple[tuple[str, str], ...] = DEFAULT_DOMAIN_DEFINITIONS
    energy_group_edges_ev: tuple[float, ...] = DEFAULT_ENERGY_GROUP_EDGES_EV
    mgxs_types: tuple[str, ...] = DEFAULT_MGXS_TYPES
    delayed_groups: tuple[int, ...] = DEFAULT_DELAYED_GROUPS
    legendre_order: int = 0


def legendre_order_for_moment(moment: str | int) -> int:
    if isinstance(moment, int):
        order = moment
    else:
        normalized = str(moment).strip().upper()
        if normalized.startswith("P"):
            normalized = normalized[1:]
        if not normalized.isdigit():
            supported = ", ".join(SUPPORTED_SCATTERING_MOMENTS)
            raise ValueError(f"Unsupported scattering moment {moment!r}. Use one of: {supported}.")
        order = int(normalized)

    if order not in SUPPORTED_LEGENDRE_ORDERS:
        supported = ", ".join(SUPPORTED_SCATTERING_MOMENTS)
        raise ValueError(f"Unsupported scattering moment order {order}. Use one of: {supported}.")
    return order


def scattering_moment_for_legendre_order(order: int) -> str:
    validated_order = legendre_order_for_moment(order)
    return f"P{validated_order}"


def energy_group_edges_for_group_count(group_count: int) -> tuple[float, ...]:
    if group_count == 1:
        return DEFAULT_ENERGY_GROUP_EDGES_EV
    if group_count == 2:
        return TWO_GROUP_ENERGY_GROUP_EDGES_EV
    if group_count == 4:
        return FOUR_GROUP_ENERGY_GROUP_EDGES_EV
    if group_count == 8:
        return EIGHT_GROUP_ENERGY_GROUP_EDGES_EV
    if group_count == 16:
        return CASMO_16_ENERGY_GROUP_EDGES_EV
    if group_count == 25:
        return CASMO_25_ENERGY_GROUP_EDGES_EV
    if group_count == 40:
        return CASMO_40_ENERGY_GROUP_EDGES_EV
    if group_count == 70:
        return CASMO_70_ENERGY_GROUP_EDGES_EV
    supported = ", ".join(str(value) for value in SUPPORTED_GROUP_COUNTS)
    raise ValueError(f"Unsupported group count {group_count}. Use one of: {supported}.")


def domain_mapping_from_definitions(
    domain_definitions: tuple[tuple[str, str], ...] = DEFAULT_DOMAIN_DEFINITIONS,
) -> dict[str, str]:
    mapping = {label: cell_name for label, cell_name in domain_definitions}
    if len(mapping) != len(domain_definitions):
        raise ValueError("Domain labels must be unique")
    return mapping


def _sorted_fuel_ring_names(cell_names: list[str]) -> list[str]:
    ring_entries: list[tuple[int, str]] = []
    for cell_name in cell_names:
        match = _FUEL_RING_NAME_PATTERN.match(cell_name)
        if match is None:
            continue
        ring_entries.append((int(match.group(1)), cell_name))
    return [cell_name for _, cell_name in sorted(ring_entries)]


def domain_definitions_for_core_modeling(
    model_path_or_dir: str | Path = DEFAULT_MODEL_XML_PATH,
    core_modeling: str = "supercell",
) -> tuple[tuple[str, str], ...]:
    if core_modeling == "supercell":
        return DEFAULT_DOMAIN_DEFINITIONS

    if core_modeling != "resolved":
        supported = ", ".join(SUPPORTED_CORE_MODELING)
        raise ValueError(f"Unsupported core_modeling {core_modeling!r}. Use one of: {supported}.")

    model_xml_path = resolve_model_xml_path(model_path_or_dir)
    openmc.reset_auto_ids()
    model = openmc.Model.from_model_xml(model_xml_path)
    geometry = model.geometry
    if geometry is None:
        raise ValueError(f"Loaded model at {model_xml_path} does not contain geometry")

    cells_by_name = {
        cell.name: cell
        for cell in geometry.get_all_cells().values()
        if cell.name
    }
    cell_names = sorted(cells_by_name)
    ring_names = _sorted_fuel_ring_names(cell_names)
    if not ring_names:
        raise ValueError(
            "Resolved core modeling requested, but no fuel ring cells named like 'fuel_ring_<index>' were found."
        )

    required_outer_regions = {
        "moderator": "d2o_tank",
        "reflector": "h2o_tank",
    }
    for region_label, cell_name in required_outer_regions.items():
        if cell_name not in cell_names:
            raise ValueError(
                f"Resolved core modeling requires cell {cell_name!r} for region {region_label!r}, but it was not found."
            )

    resolved_domains: list[tuple[str, str]] = []
    used_cell_names: set[str] = set()

    if "central_moderator_channel" in cell_names:
        resolved_domains.append(("core_central_moderator_channel", "central_moderator_channel"))
        used_cell_names.add("central_moderator_channel")

    for ring_name in ring_names:
        resolved_domains.append((f"core_{ring_name}", ring_name))
        used_cell_names.add(ring_name)

    # Ensure OpenMC MG mode has XS data for every material-filled core cell.
    for cell_name in cell_names:
        if cell_name in used_cell_names:
            continue
        if cell_name in required_outer_regions.values():
            continue

        cell = cells_by_name[cell_name]
        if isinstance(cell.fill, openmc.Material):
            resolved_domains.append((f"core_{cell_name}", cell_name))
            used_cell_names.add(cell_name)

    resolved_domains.extend([
        ("moderator", "d2o_tank"),
        ("reflector", "h2o_tank"),
    ])
    return tuple(resolved_domains)


def default_openmc_threads() -> int:
    return int(os.environ.get("OPENMC_THREADS", str(os.cpu_count() or 1)))


def cross_sections_path() -> str | None:
    return openmc.config.get("cross_sections")


def resolve_model_xml_path(model_path_or_dir: str | Path = DEFAULT_MODEL_XML_PATH) -> Path:
    candidate = Path(model_path_or_dir).expanduser()
    model_xml_path = candidate if candidate.name == "model.xml" else candidate / "model.xml"
    if not model_xml_path.exists():
        raise FileNotFoundError(
            f"Could not find model.xml at {model_xml_path}. Build the core model first and point the notebook at that build directory."
        )
    return model_xml_path.resolve()


def build_directories(
    model_path_or_dir: str | Path = DEFAULT_MODEL_XML_PATH,
    export_dir: str | Path | None = None,
) -> dict[str, Path]:
    model_xml_path = resolve_model_xml_path(model_path_or_dir)
    model_dir = model_xml_path.parent
    root_dir = (Path(export_dir) if export_dir is not None else model_dir / DEFAULT_EXPORT_DIRNAME).resolve()
    run_dir = root_dir / "reactor_run"
    output_dir = root_dir / "outputs"
    for path in (root_dir, run_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "model_xml_path": model_xml_path,
        "model_dir": model_dir,
        "root_dir": root_dir,
        "run_dir": run_dir,
        "output_dir": output_dir,
    }


def inspect_model_source(
    model_path_or_dir: str | Path = DEFAULT_MODEL_XML_PATH,
    domain_definitions: tuple[tuple[str, str], ...] = DEFAULT_DOMAIN_DEFINITIONS,
) -> dict[str, Any]:
    model_xml_path = resolve_model_xml_path(model_path_or_dir)
    openmc.reset_auto_ids()
    model = openmc.Model.from_model_xml(model_xml_path)
    geometry = model.geometry
    if geometry is None:
        raise ValueError(f"Loaded model at {model_xml_path} does not contain geometry")
    settings = model.settings
    cell_names = sorted(cell.name for cell in geometry.get_all_cells().values() if cell.name)
    requested_domains = domain_mapping_from_definitions(domain_definitions)
    return {
        "model_xml_path": str(model_xml_path),
        "model_directory": str(model_xml_path.parent),
        "model_name": model_xml_path.parent.name,
        "domains": {
            label: {
                "cell_name": cell_name,
                "present": cell_name in cell_names,
            }
            for label, cell_name in requested_domains.items()
        },
        "cell_count": len(cell_names),
        "available_cells": cell_names,
        "settings": {
            "run_mode": settings.run_mode,
            "particles": settings.particles,
            "batches": settings.batches,
            "inactive": settings.inactive,
            "source_present": settings.source is not None,
        },
    }


def _is_mesh_tally(tally: openmc.Tally) -> bool:
    # Plotting notebooks may persist mesh tallies in model XML; MGXS runs do not use them.
    return any("Mesh" in type(tally_filter).__name__ for tally_filter in tally.filters)


def _remove_mesh_tallies(model: openmc.Model) -> dict[str, int]:
    if model.tallies is None:
        return {
            "total": 0,
            "removed_mesh": 0,
            "retained_non_mesh": 0,
        }

    retained = openmc.Tallies()
    removed_mesh = 0
    total = len(model.tallies)
    for tally in model.tallies:
        if _is_mesh_tally(tally):
            removed_mesh += 1
            continue
        retained.append(tally)

    model.tallies = retained
    return {
        "total": total,
        "removed_mesh": removed_mesh,
        "retained_non_mesh": len(retained),
    }


def load_model_from_xml(
    model_path_or_dir: str | Path,
    config: MGXSExportConfig,
) -> tuple[openmc.Model, dict[str, Any]]:
    model_xml_path = resolve_model_xml_path(model_path_or_dir)
    openmc.reset_auto_ids()
    model = openmc.Model.from_model_xml(model_xml_path)

    if model.geometry is None:
        raise ValueError(f"Loaded model at {model_xml_path} does not contain geometry")

    settings = model.settings
    if settings.source is None:
        raise ValueError(
            f"Loaded model at {model_xml_path} does not contain a source definition. Build a runnable eigenvalue model first."
        )

    settings.run_mode = "eigenvalue"
    settings.particles = config.particles
    settings.batches = config.batches
    settings.inactive = config.inactive
    settings.max_order = legendre_order_for_moment(config.legendre_order)
    model.settings = settings

    tally_cleanup = _remove_mesh_tallies(model)

    metadata = inspect_model_source(model_xml_path, domain_definitions=config.domain_definitions)
    metadata["configured_run"] = {
        "particles": config.particles,
        "batches": config.batches,
        "inactive": config.inactive,
        "group_count": len(config.energy_group_edges_ev) - 1,
        "scattering_moment": scattering_moment_for_legendre_order(config.legendre_order),
        "legendre_order": int(config.legendre_order),
        "domain_definitions": [list(item) for item in config.domain_definitions],
        "energy_group_edges_ev": list(config.energy_group_edges_ev),
        "mgxs_types": list(config.mgxs_types),
        "delayed_groups": list(config.delayed_groups),
    }
    metadata["tally_cleanup"] = tally_cleanup
    return model, metadata


def _find_cell_by_name(geometry: openmc.Geometry, cell_name: str) -> openmc.Cell:
    matches = [cell for cell in geometry.get_all_cells().values() if cell.name == cell_name]
    if not matches:
        raise ValueError(f"Could not find a cell named {cell_name!r} in the model geometry")
    if len(matches) > 1:
        raise ValueError(f"Expected one cell named {cell_name!r}, found {len(matches)}")
    return matches[0]


def attach_mgxs_tallies(
    model: openmc.Model,
    config: MGXSExportConfig,
) -> tuple[
    mgxs.Library,
    dict[str, mgxs.Beta],
    dict[str, mgxs.DecayRate],
    dict[str, openmc.Cell],
    dict[str, Any],
]:
    geometry = model.geometry
    if geometry is None:
        raise ValueError("Model must contain geometry before MGXS tallies can be attached")

    domain_mapping = domain_mapping_from_definitions(config.domain_definitions)
    domains = {
        label: _find_cell_by_name(geometry, cell_name)
        for label, cell_name in domain_mapping.items()
    }
    energy_groups = mgxs.EnergyGroups(np.asarray(config.energy_group_edges_ev, dtype=float))

    library = mgxs.Library(
        geometry=geometry,
        by_nuclide=False,
        mgxs_types=list(config.mgxs_types),
        name="region-mgxs",
    )
    library.energy_groups = energy_groups
    library.domain_type = "cell"
    library.domains = list(domains.values())
    library.legendre_order = legendre_order_for_moment(config.legendre_order)
    library.build_library()

    tallies = openmc.Tallies()
    if hasattr(library, "add_to_tallies"):
        library.add_to_tallies(tallies, merge=True)
    else:
        library.add_to_tallies_file(tallies, merge=True)

    beta_by_domain = {
        label: mgxs.Beta(
            domain=domain,
            domain_type="cell",
            energy_groups=energy_groups,
            delayed_groups=list(config.delayed_groups),
            name=f"{label}-beta",
        )
        for label, domain in domains.items()
    }
    decay_rate_by_domain = {
        label: mgxs.DecayRate(
            domain=domain,
            domain_type="cell",
            energy_groups=energy_groups,
            delayed_groups=list(config.delayed_groups),
            name=f"{label}-decay-rate",
        )
        for label, domain in domains.items()
    }
    for kinetics_xs in [*beta_by_domain.values(), *decay_rate_by_domain.values()]:
        for tally in kinetics_xs.tallies.values():
            tallies.append(tally, merge=True)

    model.tallies = tallies
    return library, beta_by_domain, decay_rate_by_domain, domains, {
        "domains": {
            label: {
                "domain_id": domain.id,
                "cell_name": domain.name,
            }
            for label, domain in domains.items()
        },
        "tally_count": len(tallies),
        "energy_groups": list(config.energy_group_edges_ev),
        "legendre_order": int(config.legendre_order),
        "scattering_moment": scattering_moment_for_legendre_order(config.legendre_order),
    }


def _uncertain_value(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if hasattr(value, "n") and hasattr(value, "s"):
        return {
            "mean": float(value.n),
            "std_dev": float(value.s),
        }
    return {
        "mean": float(value),
        "std_dev": 0.0,
    }


def _serialise_array(value: Any) -> float | list[Any]:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.tolist()


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _vectorise_delayed_xs(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    squeezed = np.squeeze(array)
    if squeezed.ndim == 0:
        return np.asarray([float(squeezed)], dtype=float)
    if squeezed.ndim > 1:
        return squeezed.reshape(-1)
    return squeezed


def _beta_matrix(values: Any, energy_group_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    squeezed = np.squeeze(array)
    if squeezed.ndim == 0:
        return np.asarray([[float(squeezed)]], dtype=float)
    if squeezed.ndim == 1:
        return squeezed.reshape(-1, 1)
    if squeezed.ndim == 2:
        if squeezed.shape[1] == energy_group_count:
            return squeezed
        if squeezed.shape[0] == energy_group_count:
            return squeezed.T
    raise ValueError(f"Unsupported beta shape {squeezed.shape} for {energy_group_count} energy groups")


def _mgxs_rows(xs_type: str, mean: Any, std_dev: Any) -> list[dict[str, float | int | str]]:
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std_dev, dtype=float)

    if mean_array.ndim == 0:
        return [
            {
                "xs_type": xs_type,
                "group": 1,
                "group_out": 1,
                "legendre_order": 0,
                "mean": float(mean_array),
                "std_dev": float(std_array),
            }
        ]

    if mean_array.ndim == 1:
        return [
            {
                "xs_type": xs_type,
                "group": group_index + 1,
                "group_out": group_index + 1,
                "legendre_order": 0,
                "mean": float(mean_value),
                "std_dev": float(std_value),
            }
            for group_index, (mean_value, std_value) in enumerate(zip(mean_array, std_array, strict=True))
        ]

    if mean_array.ndim == 2:
        rows: list[dict[str, float | int | str]] = []
        for group_index in range(mean_array.shape[0]):
            for group_out_index in range(mean_array.shape[1]):
                rows.append(
                    {
                        "xs_type": xs_type,
                        "group": group_index + 1,
                        "group_out": group_out_index + 1,
                        "legendre_order": 0,
                        "mean": float(mean_array[group_index, group_out_index]),
                        "std_dev": float(std_array[group_index, group_out_index]),
                    }
                )
        return rows

    if mean_array.ndim == 3:
        rows: list[dict[str, float | int | str]] = []
        # Higher-order scattering moments are exported with a Legendre axis.
        for legendre_order in range(mean_array.shape[0]):
            for group_index in range(mean_array.shape[1]):
                for group_out_index in range(mean_array.shape[2]):
                    rows.append(
                        {
                            "xs_type": xs_type,
                            "group": group_index + 1,
                            "group_out": group_out_index + 1,
                            "legendre_order": legendre_order,
                            "mean": float(mean_array[legendre_order, group_index, group_out_index]),
                            "std_dev": float(std_array[legendre_order, group_index, group_out_index]),
                        }
                    )
        return rows

    raise ValueError(f"Unsupported MGXS rank {mean_array.ndim} for {xs_type!r}")


def load_mgxs_results(
    statepoint_path: str | Path,
    library: mgxs.Library,
    beta_by_domain: dict[str, mgxs.Beta],
    decay_rate_by_domain: dict[str, mgxs.DecayRate],
    domains: dict[str, openmc.Cell],
    config: MGXSExportConfig,
    model_metadata: dict[str, Any] | None = None,
    export_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    statepoint_path = Path(statepoint_path).resolve()
    energy_group_count = len(config.energy_group_edges_ev) - 1
    with openmc.StatePoint(statepoint_path) as statepoint:
        library.load_from_statepoint(statepoint)
        for beta in beta_by_domain.values():
            beta.load_from_statepoint(statepoint)
        for decay_rate in decay_rate_by_domain.values():
            decay_rate.load_from_statepoint(statepoint)

        all_group_constant_rows: list[dict[str, float | int | str]] = []
        all_delayed_rows: list[dict[str, float | int | str]] = []
        domain_results: dict[str, Any] = {}
        for label, domain in domains.items():
            mgxs_data: dict[str, dict[str, Any]] = {}
            group_constant_rows: list[dict[str, float | int | str]] = []
            for mgxs_type in config.mgxs_types:
                mgxs_object = library.get_mgxs(domain, mgxs_type)
                mean = mgxs_object.get_xs(value="mean")
                std_dev = mgxs_object.get_xs(value="std_dev")
                mgxs_data[mgxs_type] = {
                    "mean": _serialise_array(mean),
                    "std_dev": _serialise_array(std_dev),
                }
                rows = _mgxs_rows(mgxs_type, mean, std_dev)
                for row in rows:
                    row["domain"] = label
                group_constant_rows.extend(rows)
                all_group_constant_rows.extend(rows)

            beta_mean = _beta_matrix(
                beta_by_domain[label].get_xs(value="mean", delayed_groups="all"),
                energy_group_count,
            )
            beta_std = _beta_matrix(
                beta_by_domain[label].get_xs(value="std_dev", delayed_groups="all"),
                energy_group_count,
            )
            decay_mean = _vectorise_delayed_xs(
                decay_rate_by_domain[label].get_xs(value="mean", delayed_groups="all")
            )
            decay_std = _vectorise_delayed_xs(
                decay_rate_by_domain[label].get_xs(value="std_dev", delayed_groups="all")
            )
            beta_total = float(np.sum(beta_mean))
            beta_total_by_delayed_group = np.sum(beta_mean, axis=1)
            beta_total_by_energy_group = np.sum(beta_mean, axis=0)

            delayed_rows = [
                {
                    "domain": label,
                    "delayed_group": int(delayed_group),
                    "energy_group": energy_group_index + 1,
                    "beta": float(beta_mean[delayed_group_index, energy_group_index]),
                    "beta_std_dev": float(beta_std[delayed_group_index, energy_group_index]),
                    "decay_rate_per_s": float(decay_group_mean),
                    "decay_rate_std_dev": float(decay_group_std),
                }
                for delayed_group_index, (delayed_group, decay_group_mean, decay_group_std) in enumerate(
                    zip(config.delayed_groups, decay_mean, decay_std, strict=True)
                )
                for energy_group_index in range(energy_group_count)
            ]
            all_delayed_rows.extend(delayed_rows)

            domain_results[label] = {
                "domain": {
                    "type": "cell",
                    "name": domain.name,
                    "id": domain.id,
                },
                "group_constants": mgxs_data,
                "group_constant_rows": group_constant_rows,
                "delayed_neutrons": {
                    "beta_total": beta_total,
                    "beta_by_delayed_group": _serialise_array(beta_mean[:, 0] if energy_group_count == 1 else beta_mean),
                    "beta_std_dev_by_delayed_group": _serialise_array(
                        beta_std[:, 0] if energy_group_count == 1 else beta_std
                    ),
                    "beta_total_by_delayed_group": beta_total_by_delayed_group.tolist(),
                    "beta_total_by_energy_group": beta_total_by_energy_group.tolist(),
                    "decay_rate_per_s_by_delayed_group": decay_mean.tolist(),
                    "decay_rate_std_dev_by_delayed_group": decay_std.tolist(),
                    "beta_weighted_decay_rate_per_s": (
                        float(np.average(decay_mean, weights=beta_total_by_delayed_group)) if beta_total > 0.0 else None
                    ),
                },
                "delayed_neutron_rows": delayed_rows,
            }

        results = {
            "statepoint_path": str(statepoint_path),
            "cross_sections": cross_sections_path(),
            "config": asdict(config),
            "model": model_metadata or {},
            "run": {
                "keff": _uncertain_value(statepoint.keff),
                "reactivity_pcm": float((statepoint.keff.n - 1.0) / statepoint.keff.n * 1.0e5),
                "generation_time_s": None,
                "k_generation_by_generation": _serialise_array(getattr(statepoint, "k_generation", [])),
            },
            "domains": domain_results,
            "group_constant_rows": all_group_constant_rows,
            "delayed_neutron_rows": all_delayed_rows,
        }
        if export_paths is not None:
            results["paths"] = {name: str(path) for name, path in export_paths.items()}
        return results


def write_results(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mgxs_constants.json"
    group_constants_csv_path = output_dir / "group_constants.csv"
    delayed_neutrons_csv_path = output_dir / "delayed_neutrons.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2, sort_keys=True)

    group_constant_rows = results.get("group_constant_rows", [])
    with group_constants_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain", "xs_type", "group", "group_out", "legendre_order", "mean", "std_dev"],
        )
        writer.writeheader()
        writer.writerows(group_constant_rows)

    delayed_rows = results.get("delayed_neutron_rows", [])
    with delayed_neutrons_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "domain",
                "delayed_group",
                "energy_group",
                "beta",
                "beta_std_dev",
                "decay_rate_per_s",
                "decay_rate_std_dev",
            ],
        )
        writer.writeheader()
        writer.writerows(delayed_rows)

    return {
        "json": json_path,
        "group_constants_csv": group_constants_csv_path,
        "delayed_neutrons_csv": delayed_neutrons_csv_path,
    }


def run_mgxs_export(
    model_path_or_dir: str | Path = DEFAULT_MODEL_XML_PATH,
    config: MGXSExportConfig = MGXSExportConfig(),
    export_dir: str | Path | None = None,
    threads: int | None = None,
    openmc_exec: str | None = None,
) -> dict[str, Any]:
    configured_cross_sections = cross_sections_path()
    if not configured_cross_sections:
        raise RuntimeError(
            "Set openmc.config['cross_sections'] or OPENMC_CROSS_SECTIONS before running MGXS export."
        )

    export_paths = build_directories(model_path_or_dir, export_dir=export_dir)
    model, model_metadata = load_model_from_xml(export_paths["model_xml_path"], config)
    library, beta_by_domain, decay_rate_by_domain, domains, tally_metadata = attach_mgxs_tallies(model, config)
    model_metadata["mgxs_tallies"] = tally_metadata

    statepoint_path = model.run(
        cwd=export_paths["run_dir"],
        threads=threads or default_openmc_threads(),
        openmc_exec=resolve_openmc_exec(openmc_exec),
        export_model_xml=True,
    )

    results = load_mgxs_results(
        statepoint_path=statepoint_path,
        library=library,
        beta_by_domain=beta_by_domain,
        decay_rate_by_domain=decay_rate_by_domain,
        domains=domains,
        config=config,
        model_metadata=model_metadata,
        export_paths=export_paths,
    )
    written_files = write_results(results, export_paths["output_dir"])
    results["files"] = {name: str(path) for name, path in written_files.items()}
    return results


def run_openmc_mg_validation(
    statepoint_path: str | Path,
    config: MGXSExportConfig,
    model_path_or_dir: str | Path = DEFAULT_MODEL_XML_PATH,
    output_dir: str | Path | None = None,
    particles: int = 4000,
    batches: int = 60,
    inactive: int = 15,
    threads: int | None = None,
    openmc_exec: str | None = None,
    apply_domain_chi: bool = False,
) -> dict[str, Any]:
    if (
        "nu-scatter matrix" not in config.mgxs_types
        and "consistent nu-scatter matrix" not in config.mgxs_types
    ):
        raise ValueError(
            "OpenMC multigroup validation requires 'nu-scatter matrix' or "
            "'consistent nu-scatter matrix' in config.mgxs_types. "
            "Re-run the export with a config that includes it."
        )

    statepoint_path = Path(statepoint_path).resolve()
    if not statepoint_path.exists():
        raise FileNotFoundError(f"Could not find statepoint file at {statepoint_path}")

    model, model_metadata = load_model_from_xml(model_path_or_dir, config)
    library, beta_by_domain, decay_rate_by_domain, domains, tally_metadata = attach_mgxs_tallies(model, config)

    validation_dir = Path(output_dir).resolve() if output_dir is not None else statepoint_path.parent / "mg_mode_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    with openmc.StatePoint(statepoint_path) as statepoint:
        library.load_from_statepoint(statepoint)
        for beta in beta_by_domain.values():
            beta.load_from_statepoint(statepoint)
        for decay_rate in decay_rate_by_domain.values():
            decay_rate.load_from_statepoint(statepoint)
        mgxs_file, materials, geometry = library.create_mg_mode(apply_domain_chi=apply_domain_chi)

    mgxs_hdf5_path = validation_dir / "mgxs.h5"
    mgxs_file.export_to_hdf5(mgxs_hdf5_path)
    materials.cross_sections = str(mgxs_hdf5_path)

    settings = copy.deepcopy(model.settings)
    settings.energy_mode = "multi-group"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    # MGXS datasets produced here are single-temperature (typically 294 K).
    # Force nearest lookup so inherited CE material temperatures do not abort MG runs.
    settings.temperature = {
        "method": "nearest",
        "default": 294.0,
        "tolerance": 1.0e6,
    }
    settings.output = {"summary": False, "tallies": False}

    mg_model = openmc.Model(geometry=geometry, materials=materials, settings=settings)
    validation_statepoint_path = mg_model.run(
        cwd=validation_dir,
        threads=threads or default_openmc_threads(),
        openmc_exec=resolve_openmc_exec(openmc_exec),
        export_model_xml=True,
    )

    with openmc.StatePoint(validation_statepoint_path) as validation_statepoint:
        mg_keff = _uncertain_value(validation_statepoint.keff)

    return {
        "source_statepoint_path": str(statepoint_path),
        "statepoint_path": str(Path(validation_statepoint_path).resolve()),
        "mgxs_hdf5_path": str(mgxs_hdf5_path),
        "energy_mode": "multi-group",
        "settings": {
            "particles": particles,
            "batches": batches,
            "inactive": inactive,
            "threads": threads or default_openmc_threads(),
            "apply_domain_chi": bool(apply_domain_chi),
            "group_count": len(config.energy_group_edges_ev) - 1,
            "max_order": settings.max_order,
            "temperature": settings.temperature,
        },
        "model": {
            **model_metadata,
            "mgxs_tallies": tally_metadata,
        },
        "run": {
            "keff": mg_keff,
            "reactivity_pcm": float((mg_keff["mean"] - 1.0) / mg_keff["mean"] * 1.0e5),
        },
        "files": {
            "mgxs_hdf5": str(mgxs_hdf5_path),
            "materials_xml": str(validation_dir / "materials.xml"),
            "geometry_xml": str(validation_dir / "geometry.xml"),
            "settings_xml": str(validation_dir / "settings.xml"),
            "model_xml": str(validation_dir / "model.xml"),
            "statepoint": str(Path(validation_statepoint_path).resolve()),
        },
    }


def run_reference_mgxs_export(
    config: MGXSExportConfig = MGXSExportConfig(),
    base_dir: Path | None = None,
    threads: int | None = None,
    openmc_exec: str | None = None,
) -> dict[str, Any]:
    export_dir = base_dir if base_dir is not None else None
    return run_mgxs_export(
        model_path_or_dir=DEFAULT_MODEL_XML_PATH,
        config=config,
        export_dir=export_dir,
        threads=threads,
        openmc_exec=openmc_exec,
    )