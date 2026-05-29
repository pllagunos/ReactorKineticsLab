from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openmc
import openmc.mgxs as mgxs

from concentric_fuel import ConcentricElementParameters, build_parameter_report, validate_parameters
from ploting import resolve_openmc_exec
from reactor_geometry import ReactorTankParameters, build_reactor_model, validate_reactor_tanks


DEFAULT_MGXS_TYPES = (
    "total",
    "transport",
    "absorption",
    "diffusion-coefficient",
    "nu-fission",
    "kappa-fission",
    "chi",
    "scatter matrix",
    "inverse-velocity",
)

DEFAULT_ENERGY_GROUP_EDGES_EV = (0.0, 20.0e6)
DEFAULT_DELAYED_GROUPS = (1, 2, 3, 4, 5, 6)
DEFAULT_BUILD_DIR = Path("build") / "notebook" / "export_mgxs"


@dataclass(frozen=True)
class MGXSExportConfig:
    rod_insertion: float = 0.0
    particles: int = 16000
    batches: int = 20
    inactive: int = 5
    domain_name: str = "fuel_element"
    energy_group_edges_ev: tuple[float, ...] = DEFAULT_ENERGY_GROUP_EDGES_EV
    mgxs_types: tuple[str, ...] = DEFAULT_MGXS_TYPES
    delayed_groups: tuple[int, ...] = DEFAULT_DELAYED_GROUPS


def reference_fuel_parameters() -> ConcentricElementParameters:
    return ConcentricElementParameters(
        ring_count=7,
        ring_thickness_cm=0.5,
        coolant_gap_cm=7.0,
        inner_radius_cm=4.5,
        outer_radius_cm=50.0,
        control_rod_radius_cm=4.0,
        h_active_cm=300.0,
        lower_plenum_cm=50.0,
        upper_plenum_cm=50.0,
        fuel_density_g_per_cm3=12.2,
        fuel_enrichment_wt_pct=0.7198,
    )


def reference_tank_parameters() -> ReactorTankParameters:
    return ReactorTankParameters(
        d2o_tank_radius_cm=250.0,
        h2o_tank_radius_cm=500.0,
        h_d2o_tank_cm=600.0,
        h_h2o_tank_cm=1000.0,
    )


def reference_geometry_report() -> dict[str, Any]:
    fuel = reference_fuel_parameters()
    tanks = reference_tank_parameters()
    validate_parameters(fuel)
    validate_reactor_tanks(fuel, tanks)
    return {
        "fuel_element": build_parameter_report(fuel),
        "reactor_tanks": tanks.to_geometry_dict(),
    }


def default_openmc_threads() -> int:
    return int(os.environ.get("OPENMC_THREADS", str(os.cpu_count() or 1)))


def cross_sections_path() -> str | None:
    return openmc.config.get("cross_sections")


def build_directories(base_dir: Path | None = None) -> dict[str, Path]:
    root_dir = (base_dir or DEFAULT_BUILD_DIR).resolve()
    run_dir = root_dir / "reactor_run"
    output_dir = root_dir / "outputs"
    for path in (root_dir, run_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root_dir": root_dir,
        "run_dir": run_dir,
        "output_dir": output_dir,
    }


def build_reference_model(config: MGXSExportConfig) -> tuple[openmc.Model, dict[str, Any]]:
    openmc.reset_auto_ids()

    fuel = reference_fuel_parameters()
    tanks = reference_tank_parameters()
    validate_parameters(fuel)
    validate_reactor_tanks(fuel, tanks)

    rod_insertion = float(np.clip(config.rod_insertion, 0.0, 1.0))
    model, metadata = build_reactor_model(
        fuel,
        tanks,
        rod_insertion=rod_insertion,
    )

    active_half = 0.5 * fuel.h_active_cm
    source = openmc.IndependentSource(
        space=openmc.stats.CylindricalIndependent(
            r=openmc.stats.PowerLaw(
                fuel.inner_radius_cm,
                fuel.outer_radius_cm,
                1.0,
            ),
            phi=openmc.stats.Uniform(0.0, 2.0 * np.pi),
            z=openmc.stats.Uniform(-active_half, active_half),
        ),
        constraints={"fissionable": True},
    )

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = config.particles
    settings.batches = config.batches
    settings.inactive = config.inactive
    settings.source = source
    settings.source_rejection_fraction = 0.01
    settings.temperature = {"method": "interpolation"}
    model.settings = settings

    metadata.update(
        {
            "rod_insertion": rod_insertion,
            "particles": config.particles,
            "batches": config.batches,
            "inactive": config.inactive,
            "domain_name": config.domain_name,
            "energy_group_edges_ev": list(config.energy_group_edges_ev),
            "mgxs_types": list(config.mgxs_types),
            "delayed_groups": list(config.delayed_groups),
        }
    )
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
) -> tuple[mgxs.Library, mgxs.Beta, mgxs.DecayRate, openmc.Cell, dict[str, Any]]:
    geometry = model.geometry
    if geometry is None:
        raise ValueError("Model must contain geometry before MGXS tallies can be attached")

    domain = _find_cell_by_name(geometry, config.domain_name)
    energy_groups = mgxs.EnergyGroups(np.asarray(config.energy_group_edges_ev, dtype=float))

    library = mgxs.Library(
        geometry=geometry,
        by_nuclide=False,
        mgxs_types=list(config.mgxs_types),
        name=f"{config.domain_name}-mgxs",
    )
    library.energy_groups = energy_groups
    library.domain_type = "cell"
    library.domains = [domain]
    if hasattr(library, "correction") and "scatter matrix" in config.mgxs_types:
        library.correction = "P0"
    library.build_library()

    tallies = model.tallies if model.tallies is not None else openmc.Tallies()
    if hasattr(library, "add_to_tallies"):
        library.add_to_tallies(tallies, merge=True)
    else:
        library.add_to_tallies_file(tallies, merge=True)

    beta = mgxs.Beta(
        domain=domain,
        domain_type="cell",
        energy_groups=energy_groups,
        delayed_groups=list(config.delayed_groups),
        name=f"{config.domain_name}-beta",
    )
    decay_rate = mgxs.DecayRate(
        domain=domain,
        domain_type="cell",
        energy_groups=energy_groups,
        delayed_groups=list(config.delayed_groups),
        name=f"{config.domain_name}-decay-rate",
    )
    for kinetics_xs in (beta, decay_rate):
        for tally in kinetics_xs.tallies.values():
            tallies.append(tally, merge=True)

    model.tallies = tallies
    return library, beta, decay_rate, domain, {
        "domain_id": domain.id,
        "domain_name": domain.name,
        "tally_count": len(tallies),
        "energy_groups": list(config.energy_group_edges_ev),
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


def _mgxs_rows(xs_type: str, mean: Any, std_dev: Any) -> list[dict[str, float | int | str]]:
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std_dev, dtype=float)

    if mean_array.ndim == 0:
        return [
            {
                "xs_type": xs_type,
                "group": 1,
                "group_out": 1,
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
                        "mean": float(mean_array[group_index, group_out_index]),
                        "std_dev": float(std_array[group_index, group_out_index]),
                    }
                )
        return rows

    raise ValueError(f"Unsupported MGXS rank {mean_array.ndim} for {xs_type!r}")


def load_mgxs_results(
    statepoint_path: str | Path,
    library: mgxs.Library,
    beta: mgxs.Beta,
    decay_rate: mgxs.DecayRate,
    domain: openmc.Cell,
    config: MGXSExportConfig,
    model_metadata: dict[str, Any] | None = None,
    export_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    statepoint_path = Path(statepoint_path).resolve()
    with openmc.StatePoint(statepoint_path) as statepoint:
        library.load_from_statepoint(statepoint)
        beta.load_from_statepoint(statepoint)
        decay_rate.load_from_statepoint(statepoint)

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
            group_constant_rows.extend(_mgxs_rows(mgxs_type, mean, std_dev))

        beta_mean = _vectorise_delayed_xs(beta.get_xs(value="mean", delayed_groups="all"))
        beta_std = _vectorise_delayed_xs(beta.get_xs(value="std_dev", delayed_groups="all"))
        decay_mean = _vectorise_delayed_xs(decay_rate.get_xs(value="mean", delayed_groups="all"))
        decay_std = _vectorise_delayed_xs(decay_rate.get_xs(value="std_dev", delayed_groups="all"))
        beta_total = float(np.sum(beta_mean))

        delayed_rows = [
            {
                "delayed_group": int(delayed_group),
                "beta": float(beta_group_mean),
                "beta_std_dev": float(beta_group_std),
                "decay_rate_per_s": float(decay_group_mean),
                "decay_rate_std_dev": float(decay_group_std),
            }
            for delayed_group, beta_group_mean, beta_group_std, decay_group_mean, decay_group_std in zip(
                config.delayed_groups,
                beta_mean,
                beta_std,
                decay_mean,
                decay_std,
                strict=True,
            )
        ]

        results = {
            "statepoint_path": str(statepoint_path),
            "cross_sections": cross_sections_path(),
            "domain": {
                "type": "cell",
                "name": domain.name,
                "id": domain.id,
            },
            "config": asdict(config),
            "model": model_metadata or {},
            "run": {
                "keff": _uncertain_value(statepoint.keff),
                "reactivity_pcm": float((statepoint.keff.n - 1.0) / statepoint.keff.n * 1.0e5),
                "generation_time_s": None,
                "k_generation_by_generation": _serialise_array(getattr(statepoint, "k_generation", [])),
            },
            "group_constants": mgxs_data,
            "group_constant_rows": group_constant_rows,
            "delayed_neutrons": {
                "beta_total": beta_total,
                "beta_by_group": beta_mean.tolist(),
                "beta_std_dev_by_group": beta_std.tolist(),
                "decay_rate_per_s_by_group": decay_mean.tolist(),
                "decay_rate_std_dev_by_group": decay_std.tolist(),
                "beta_weighted_decay_rate_per_s": (
                    float(np.average(decay_mean, weights=beta_mean)) if beta_total > 0.0 else None
                ),
            },
            "delayed_neutron_rows": delayed_rows,
        }
        if export_paths is not None:
            results["paths"] = {name: str(path) for name, path in export_paths.items()}
        return results


def write_results(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "one_group_constants.json"
    group_constants_csv_path = output_dir / "group_constants.csv"
    delayed_neutrons_csv_path = output_dir / "delayed_neutrons.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2, sort_keys=True)

    group_constant_rows = results.get("group_constant_rows", [])
    with group_constants_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["xs_type", "group", "group_out", "mean", "std_dev"])
        writer.writeheader()
        writer.writerows(group_constant_rows)

    delayed_rows = results.get("delayed_neutron_rows", [])
    with delayed_neutrons_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "delayed_group",
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


def run_reference_mgxs_export(
    config: MGXSExportConfig = MGXSExportConfig(),
    base_dir: Path | None = None,
    threads: int | None = None,
    openmc_exec: str | None = None,
) -> dict[str, Any]:
    configured_cross_sections = cross_sections_path()
    if not configured_cross_sections:
        raise RuntimeError(
            "Set openmc.config['cross_sections'] or OPENMC_CROSS_SECTIONS before running MGXS export."
        )

    export_paths = build_directories(base_dir)
    model, model_metadata = build_reference_model(config)
    model_metadata.pop("materials", None)
    library, beta, decay_rate, domain, tally_metadata = attach_mgxs_tallies(model, config)
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
        beta=beta,
        decay_rate=decay_rate,
        domain=domain,
        config=config,
        model_metadata=model_metadata,
        export_paths=export_paths,
    )
    written_files = write_results(results, export_paths["output_dir"])
    results["files"] = {name: str(path) for name, path in written_files.items()}
    return results