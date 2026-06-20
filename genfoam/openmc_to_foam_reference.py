from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import openmc
from MultiGroupXS import MultiGroupXS, MultiGroupXSManager

from openmc_to_genfoam_xs import (
    DEFAULT_OUTPUT_SUBDIR,
    _adapt_zone_data,
    build_nuclear_data_text_from_export,
    generate_genfoam_xs,
)
from prepare_concentric_case import DEFAULT_MGXS_EXPORT_DIR, DEFAULT_OUTPUT_DIR


DEFAULT_REFERENCE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "openmc_to_foam_reference"
DEFAULT_OPENMC_TO_FOAM_OUTPUT_SUBDIR = "openmc_to_foam_xs"
DEFAULT_FOCUS_FIELDS = ("sigmaRemoval", "scatteringMatrixP0")
CM_TO_M = 1.0e-2
EV_TO_J = 1.602176487e-19


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_run_value(export: dict[str, Any], key: str, fallback: int) -> int:
    configured_run = export.get("model", {}).get("configured_run", {})
    config = export.get("config", {})
    value = configured_run.get(key, config.get(key, fallback))
    return int(value)


def _set_scattering_order(mgxs_cell: Any, legendre_order: int | None) -> None:
    if legendre_order is None:
        return
    mgxs_cell.scatteringMatrixNumber = legendre_order + 1
    mgxs_cell.scatterMatrixXS.legendre_order = legendre_order
    mgxs_cell.scatterMatrixXS.correction = None
    mgxs_cell.scatterMatrixXS.formulation = "consistent"


def _build_reference_manager(
    model: openmc.Model,
    export: dict[str, Any],
    legendre_order: int | None,
) -> MultiGroupXSManager:
    geometry = model.geometry
    if geometry is None:
        raise ValueError("Loaded OpenMC model does not contain geometry")

    cells_by_name = {
        cell.name: cell
        for cell in geometry.get_all_cells().values()
        if cell.name
    }
    energy_groups = openmc.mgxs.EnergyGroups(export["config"]["energy_group_edges_ev"])
    delayed_groups = list(export["config"]["delayed_groups"])
    manager = MultiGroupXSManager(geometry.root_universe, energy_groups, delayed_groups)
    for mgxs_cell in manager.mgxsList:
        _set_scattering_order(mgxs_cell, legendre_order)

    for label, cell_name in export["config"]["domain_definitions"]:
        if cell_name not in cells_by_name:
            raise ValueError(f"Could not find cell {cell_name!r} for region {label!r} in the OpenMC model")
        mgxs_cell = MultiGroupXS(
            cells_by_name[cell_name],
            energy_groups,
            delayed_groups,
            fuelFraction=1.0 if label.startswith("core_fuel_ring_") else 0.0,
            genfoamName=label,
        )
        _set_scattering_order(mgxs_cell, legendre_order)
        manager.AddDomain(mgxs_cell)
    return manager


def _matrix_p0(scatter_xs: Any) -> list[list[float]]:
    scatter = np.asarray(scatter_xs, dtype=float)
    if scatter.ndim == 3:
        p0 = scatter[:, :, 0]
    elif scatter.ndim == 2:
        p0 = scatter
    else:
        raise ValueError(f"Unsupported scatter XS shape for P0 extraction: {scatter.shape}")
    return [[float(value) * 100.0 for value in row] for row in p0]


def _normalize_integral_flux(integral_flux: Any, master_flux: Any) -> list[float]:
    normalized: list[float] = []
    for flux, flux_master in zip(integral_flux.mean, master_flux.mean, strict=True):
        denominator = float(flux_master[0][0])
        numerator = float(flux[0][0])
        normalized.append(0.0 if denominator == 0.0 else numerator / denominator)
    return normalized[::-1]


def _extract_raw_zone_data(mgxs_cell: Any, master_flux: Any, master_lambda: list[float]) -> dict[str, Any]:
    scattering_matrix_p0 = _matrix_p0(mgxs_cell.scatterMatrixXS.get_xs())
    sigma_removal = [
        float(total - scattering_matrix_p0[group_index][group_index] / 100.0) / CM_TO_M
        for group_index, total in enumerate(mgxs_cell.totalXS.get_xs())
    ]
    lambda_values = [float(value) for value in np.asarray(mgxs_cell.Lambda.get_xs(), dtype=float).reshape(-1)]
    if not any(value > 0.0 and np.isfinite(value) for value in lambda_values):
        lambda_values = [float(value) for value in master_lambda]

    return {
        "cell_name": mgxs_cell.domain.name,
        "fuelFraction": float(mgxs_cell.fuelFraction),
        "IV": [float(value) / CM_TO_M for value in mgxs_cell.inverseVelocity.get_xs()],
        "D": [float(value) * CM_TO_M for value in mgxs_cell.diffusionCoefficient.get_xs()],
        "nuSigmaEff": [float(value) / CM_TO_M for value in mgxs_cell.nuFissionXS.get_xs()],
        "sigmaPow": [float(value) * EV_TO_J / CM_TO_M for value in mgxs_cell.kappaFissionXS.get_xs()],
        "scatteringMatrixP0": scattering_matrix_p0,
        "sigmaRemoval": sigma_removal,
        "chiPrompt": [float(value) for value in mgxs_cell.chiPrompt.get_xs()],
        "chiDelayed": [float(value) for value in mgxs_cell.chiDelayed.get_xs()[0]],
        "Beta": [float(value[0]) for value in mgxs_cell.beta.get_xs()],
        "lambda": lambda_values,
        "discFactor": [1.0] * mgxs_cell.energyGroups.num_groups,
        "integralFlux": _normalize_integral_flux(mgxs_cell.integralFlux, master_flux),
    }


def generate_openmc_to_foam_reference(
    mgxs_export_dir: Path,
    output_dir: Path,
    particles: int | None = None,
    batches: int | None = None,
    inactive: int | None = None,
    threads: int | None = None,
    legendre_order: int | None = None,
) -> dict[str, Any]:
    mgxs_export_dir = mgxs_export_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    export = _load_json(mgxs_export_dir / "outputs" / "mgxs_constants.json")
    colocated_model_xml = mgxs_export_dir / "reactor_run" / "model.xml"
    configured_model_xml = Path(
        export.get("model", {}).get("model_xml_path", colocated_model_xml)
    )
    model_xml_path = (
        colocated_model_xml
        if colocated_model_xml.is_file()
        else configured_model_xml
    )
    if not model_xml_path.exists():
        raise FileNotFoundError(f"Could not find OpenMC model XML at {model_xml_path}")

    openmc.reset_auto_ids()
    model = openmc.Model.from_model_xml(model_xml_path)
    settings = model.settings
    settings.run_mode = "eigenvalue"
    settings.particles = particles if particles is not None else _configured_run_value(export, "particles", 16000)
    settings.batches = batches if batches is not None else _configured_run_value(export, "batches", 20)
    settings.inactive = inactive if inactive is not None else _configured_run_value(export, "inactive", 5)
    model.settings = settings
    resolved_threads = threads if threads is not None else max(1, os.cpu_count() or 1)

    manager = _build_reference_manager(model, export, legendre_order=legendre_order)
    tallies = openmc.Tallies()
    manager.AddToTallies(tallies)
    model.tallies = tallies

    run_dir = output_dir / "reactor_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        statepoint_path = Path(model.run(output=False, threads=resolved_threads)).resolve()
    finally:
        os.chdir(previous_cwd)

    raw_nuclear_data_path = output_dir / "nuclearData.openmcToFoam"
    raw_nuclear_data_error: str | None = None
    with openmc.StatePoint(statepoint_path) as statepoint:
        manager.LoadFromStatepoint(statepoint)
        os.chdir(output_dir)
        try:
            try:
                manager.PrintForGeNFoam(raw_nuclear_data_path.name)
            except Exception as exc:
                raw_nuclear_data_error = f"{type(exc).__name__}: {exc}"
                if raw_nuclear_data_path.exists():
                    raw_nuclear_data_path.unlink()
        finally:
            os.chdir(previous_cwd)
        keff = {"mean": float(statepoint.keff.n), "std_dev": float(statepoint.keff.s)}
        prompt_generation_time_s = float(manager.masterMGXSOneGroup.GetPromptGenerationLifetime()[0])

    master_flux = manager.masterMGXS.integralFlux
    master_lambda = [
        float(value)
        for value in np.asarray(manager.masterMGXSOneGroup.Lambda.get_xs(), dtype=float).reshape(-1)
    ]
    domain_order = [label for label, _ in export["config"]["domain_definitions"]]
    raw_zones = {
        mgxs_cell.genfoamName: _extract_raw_zone_data(
            mgxs_cell=mgxs_cell,
            master_flux=master_flux,
            master_lambda=master_lambda,
        )
        for mgxs_cell in manager.mgxsList[2:]
    }
    adapted_zones = {
        zone_name: _adapt_zone_data(zone_name, raw_zone_data)
        for zone_name, raw_zone_data in raw_zones.items()
    }
    sanitization_summary = {
        zone_name: {
            field_name: indices
            for field_name, indices in zone_data.get("metadata", {}).get("sanitization", {}).items()
            if indices
        }
        for zone_name, zone_data in adapted_zones.items()
    }
    sanitization_summary = {
        zone_name: field_changes
        for zone_name, field_changes in sanitization_summary.items()
        if field_changes
    }

    payload = {
        "source": {
            "mgxs_export_dir": str(mgxs_export_dir),
            "model_xml_path": str(model_xml_path.resolve()),
            "tool": "openmcToFoam.MultiGroupXS",
        },
        "config": {
            "group_count": len(export["config"]["energy_group_edges_ev"]) - 1,
            "energy_group_edges_ev": export["config"]["energy_group_edges_ev"],
            "precursor_group_count": len(export["config"]["delayed_groups"]),
            "delayed_groups": export["config"]["delayed_groups"],
            "domain_order": domain_order,
            "requested_legendre_order": (
                int(legendre_order)
                if legendre_order is not None
                else None
            ),
            "effective_scattering_order": (
                int(legendre_order)
                if legendre_order is not None
                else 5
            ),
            "scatter_correction": None,
            "scatter_formulation": "consistent",
            "scatter_matrix_source": "openmcToFoam ScatterMatrixXS(nu=True)",
            "kinetics_group_count": 1,
        },
        "reference_run": {
            "statepoint_path": str(statepoint_path),
            "raw_nuclear_data_path": str(raw_nuclear_data_path) if raw_nuclear_data_path.exists() else None,
            "raw_nuclear_data_error": raw_nuclear_data_error,
            "keff": keff,
            "prompt_generation_time_s": prompt_generation_time_s,
            "particles": int(model.settings.particles),
            "batches": int(model.settings.batches),
            "inactive": int(model.settings.inactive),
            "threads": int(resolved_threads),
        },
        "raw_zones": raw_zones,
        "zones": adapted_zones,
        "notes": [
            "Reference vectors come from the installed openmcToFoam MultiGroupXS tool.",
            "Adapted vectors reuse the current local sanitization path so source-value differences stay visible.",
            "Without --openmc-to-foam-legendre-order, openmcToFoam keeps its historical default P5 scattering treatment.",
            "If MultiGroupXS.PrintForGeNFoam cannot print the selected scatter layout, the comparison still proceeds from the loaded MGXS values.",
        ],
        "sanitization_summary": sanitization_summary,
    }

    adapted_nuclear_data_text = build_nuclear_data_text_from_export(payload, domain_order)
    adapted_nuclear_data_path = output_dir / "nuclearData.genfoam"
    raw_vectors_path = output_dir / "raw_vectors.json"
    adapted_vectors_path = output_dir / "adapted_vectors.json"
    summary_path = output_dir / "summary.json"

    _write_text(adapted_nuclear_data_path, adapted_nuclear_data_text)
    _write_json(raw_vectors_path, raw_zones)
    _write_json(adapted_vectors_path, adapted_zones)
    summary = {
        "source": payload["source"],
        "config": payload["config"],
        "reference_run": payload["reference_run"],
        "sanitization_summary": sanitization_summary,
        "files": {
            "raw_vectors": str(raw_vectors_path),
            "adapted_vectors": str(adapted_vectors_path),
            "raw_nuclear_data": str(raw_nuclear_data_path) if raw_nuclear_data_path.exists() else None,
            "adapted_nuclear_data": str(adapted_nuclear_data_path),
        },
    }
    _write_json(summary_path, summary)

    payload["files"] = summary["files"] | {"summary": str(summary_path)}
    return payload


def _flatten_values(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for value in values:
            flattened.extend(_flatten_values(value))
        return flattened
    return [float(values)]


def _enumerate_entries(values: Any, prefix: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], float]]:
    if isinstance(values, list):
        entries: list[tuple[tuple[int, ...], float]] = []
        for index, value in enumerate(values, start=1):
            entries.extend(_enumerate_entries(value, (*prefix, index)))
        return entries
    return [(prefix, float(values))]


def _shape_of(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    if not values:
        return [0]
    return [len(values), *_shape_of(values[0])]


def _compare_values(reference: Any, current: Any) -> dict[str, Any]:
    reference_flat = _flatten_values(reference)
    current_flat = _flatten_values(current)
    if len(reference_flat) != len(current_flat):
        return {
            "length_mismatch": {
                "reference": len(reference_flat),
                "current": len(current_flat),
                "reference_shape": _shape_of(reference),
                "current_shape": _shape_of(current),
            }
        }
    ref = np.asarray(reference_flat, dtype=float)
    cur = np.asarray(current_flat, dtype=float)
    abs_diff = np.abs(ref - cur)
    denom = np.maximum(np.abs(ref), 1.0e-16)
    rel_diff = abs_diff / denom
    max_index = int(abs_diff.argmax()) if abs_diff.size else 0
    return {
        "reference_shape": _shape_of(reference),
        "current_shape": _shape_of(current),
        "max_abs_diff": float(abs_diff.max(initial=0.0)),
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "max_rel_diff": float(rel_diff.max(initial=0.0)),
        "max_abs_diff_index": max_index + 1 if abs_diff.size else None,
        "reference_at_max_abs_diff": float(ref[max_index]) if abs_diff.size else None,
        "current_at_max_abs_diff": float(cur[max_index]) if abs_diff.size else None,
    }


def _compare_scalar(reference: float, current: float) -> dict[str, Any]:
    abs_diff = abs(float(reference) - float(current))
    denom = max(abs(float(reference)), 1.0e-16)
    return {
        "reference": float(reference),
        "current": float(current),
        "abs_diff": abs_diff,
        "rel_diff": abs_diff / denom,
    }


def _top_entry_differences(
    reference: Any,
    current: Any,
    limit: int,
) -> list[dict[str, Any]]:
    reference_entries = _enumerate_entries(reference)
    current_entries = _enumerate_entries(current)
    if len(reference_entries) != len(current_entries):
        return [{
            "length_mismatch": {
                "reference": len(reference_entries),
                "current": len(current_entries),
                "reference_shape": _shape_of(reference),
                "current_shape": _shape_of(current),
            }
        }]

    differences: list[dict[str, Any]] = []
    for (reference_index, reference_value), (current_index, current_value) in zip(
        reference_entries,
        current_entries,
        strict=True,
    ):
        if reference_index != current_index:
            raise ValueError("Reference and current entry indices diverged during comparison")
        abs_diff = abs(reference_value - current_value)
        rel_diff = abs_diff / max(abs(reference_value), 1.0e-16)
        differences.append({
            "index": list(reference_index),
            "reference": reference_value,
            "current": current_value,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
        })

    return sorted(differences, key=lambda item: item["abs_diff"], reverse=True)[:limit]


def _largest_differences(
    comparison_by_zone: dict[str, dict[str, dict[str, Any]]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for zone_name, fields in comparison_by_zone.items():
        for field_name, metrics in fields.items():
            entries.append({
                "zone": zone_name,
                "field": field_name,
                "max_abs_diff": float(metrics.get("max_abs_diff", 0.0)),
                "max_rel_diff": float(metrics.get("max_rel_diff", 0.0)),
            })
    return sorted(entries, key=lambda item: item["max_abs_diff"], reverse=True)[:limit]


def _focused_entry_differences(
    reference_zones: dict[str, dict[str, Any]],
    current_zones: dict[str, dict[str, Any]],
    focus_fields: tuple[str, ...],
    zone_filter: tuple[str, ...] | None,
    limit: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    selected_zones = zone_filter or tuple(reference_zones.keys())
    focused: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for zone_name in selected_zones:
        if zone_name not in reference_zones or zone_name not in current_zones:
            continue
        zone_entries = {
            field_name: _top_entry_differences(
                reference_zones[zone_name][field_name],
                current_zones[zone_name][field_name],
                limit=limit,
            )
            for field_name in focus_fields
            if field_name in reference_zones[zone_name] and field_name in current_zones[zone_name]
        }
        zone_entries = {
            field_name: entries
            for field_name, entries in zone_entries.items()
            if entries
        }
        if zone_entries:
            focused[zone_name] = zone_entries
    return focused


def run_comparison(
    mgxs_export_dir: Path,
    output_dir: Path,
    rerun_local_mgxs: bool,
    local_particles: int | None,
    local_batches: int | None,
    local_inactive: int | None,
    local_legendre_order: int | None,
    openmc_to_foam_particles: int | None,
    openmc_to_foam_batches: int | None,
    openmc_to_foam_inactive: int | None,
    openmc_to_foam_legendre_order: int | None,
    focus_fields: tuple[str, ...],
    focus_zones: tuple[str, ...] | None,
    focus_limit: int,
    threads: int | None,
) -> dict[str, Any]:
    current_output_dir = output_dir / DEFAULT_OUTPUT_SUBDIR
    current_payload = generate_genfoam_xs(
        mgxs_export_dir=mgxs_export_dir,
        output_dir=current_output_dir,
        rerun_mgxs=rerun_local_mgxs,
        particles=local_particles,
        batches=local_batches,
        inactive=local_inactive,
        legendre_order=local_legendre_order,
        threads=threads,
    )

    openmc_to_foam_output_dir = output_dir / DEFAULT_OPENMC_TO_FOAM_OUTPUT_SUBDIR
    reference_payload = generate_openmc_to_foam_reference(
        mgxs_export_dir=mgxs_export_dir,
        output_dir=openmc_to_foam_output_dir,
        particles=openmc_to_foam_particles,
        batches=openmc_to_foam_batches,
        inactive=openmc_to_foam_inactive,
        threads=threads,
        legendre_order=openmc_to_foam_legendre_order,
    )

    fields = (
        "IV",
        "D",
        "nuSigmaEff",
        "sigmaPow",
        "sigmaRemoval",
        "chiPrompt",
        "chiDelayed",
        "Beta",
        "lambda",
        "integralFlux",
        "scatteringMatrixP0",
    )
    raw_comparison: dict[str, dict[str, dict[str, Any]]] = {}
    adapted_comparison: dict[str, dict[str, dict[str, Any]]] = {}
    for zone_name in reference_payload["raw_zones"]:
        raw_comparison[zone_name] = {
            field_name: _compare_values(
                reference_payload["raw_zones"][zone_name][field_name],
                current_payload["raw_zones"][zone_name][field_name],
            )
            for field_name in fields
        }
        adapted_comparison[zone_name] = {
            field_name: _compare_values(
                reference_payload["zones"][zone_name][field_name],
                current_payload["zones"][zone_name][field_name],
            )
            for field_name in fields
        }

    comparison = {
        "mgxs_export_dir": str(mgxs_export_dir),
        "notes": [
            "The current path is the local MGXS-export-derived writer.",
            "The reference path is the installed openmcToFoam MultiGroupXS workflow.",
            "Raw comparison isolates extraction differences; adapted comparison keeps the current sanitization in both paths.",
        ],
        "current_writer": {
            "summary_path": current_payload["files"]["summary"],
            "reference_run": current_payload["reference_run"],
        },
        "openmc_to_foam_reference": {
            "summary_path": reference_payload["files"]["summary"],
            "reference_run": reference_payload["reference_run"],
        },
        "run_level_comparison": {
            "prompt_generation_time_s": _compare_scalar(
                reference_payload["reference_run"]["prompt_generation_time_s"],
                current_payload["reference_run"]["prompt_generation_time_s"],
            ),
            "keff_mean": _compare_scalar(
                reference_payload["reference_run"]["keff"]["mean"],
                current_payload["reference_run"]["keff"]["mean"],
            ) if isinstance(current_payload["reference_run"].get("keff"), dict) else None,
        },
        "raw_zone_comparison": raw_comparison,
        "adapted_zone_comparison": adapted_comparison,
        "largest_raw_differences": _largest_differences(raw_comparison),
        "largest_adapted_differences": _largest_differences(adapted_comparison),
        "focused_raw_entry_differences": _focused_entry_differences(
            reference_payload["raw_zones"],
            current_payload["raw_zones"],
            focus_fields=focus_fields,
            zone_filter=focus_zones,
            limit=focus_limit,
        ),
        "focused_adapted_entry_differences": _focused_entry_differences(
            reference_payload["zones"],
            current_payload["zones"],
            focus_fields=focus_fields,
            zone_filter=focus_zones,
            limit=focus_limit,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.json"
    summary_path = output_dir / "summary.json"
    _write_json(comparison_path, comparison)
    summary = {
        "mgxs_export_dir": str(mgxs_export_dir),
        "comparison_path": str(comparison_path),
        "current_summary_path": current_payload["files"]["summary"],
        "openmc_to_foam_summary_path": reference_payload["files"]["summary"],
        "largest_raw_differences": comparison["largest_raw_differences"][:10],
        "largest_adapted_differences": comparison["largest_adapted_differences"][:10],
        "focused_raw_entry_differences": comparison["focused_raw_entry_differences"],
        "focused_adapted_entry_differences": comparison["focused_adapted_entry_differences"],
    }
    _write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the installed openmcToFoam MultiGroupXS path and compare its zone-wise MGXS "
            "against the local MGXS-export-derived GeN-Foam writer."
        )
    )
    parser.add_argument("--mgxs-export-dir", type=Path, default=DEFAULT_MGXS_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REFERENCE_OUTPUT_DIR)
    parser.add_argument("--rerun-local-mgxs", action="store_true")
    parser.add_argument("--local-particles", type=int, default=None)
    parser.add_argument("--local-batches", type=int, default=None)
    parser.add_argument("--local-inactive", type=int, default=None)
    parser.add_argument("--local-legendre-order", type=int, default=None)
    parser.add_argument("--openmc-to-foam-particles", type=int, default=None)
    parser.add_argument("--openmc-to-foam-batches", type=int, default=None)
    parser.add_argument("--openmc-to-foam-inactive", type=int, default=None)
    parser.add_argument(
        "--openmc-to-foam-legendre-order",
        type=int,
        default=None,
        help="Override openmcToFoam's historical default P5 scattering treatment for an apples-to-apples comparison.",
    )
    parser.add_argument(
        "--focus-fields",
        nargs="+",
        default=list(DEFAULT_FOCUS_FIELDS),
        help="Fields for entry-level diff reporting. Defaults to sigmaRemoval and scatteringMatrixP0.",
    )
    parser.add_argument(
        "--focus-zones",
        nargs="+",
        default=None,
        help="Optional subset of zones for entry-level diff reporting.",
    )
    parser.add_argument(
        "--focus-limit",
        type=int,
        default=8,
        help="Maximum entry-level differences to keep per zone and field.",
    )
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_comparison(
        mgxs_export_dir=args.mgxs_export_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        rerun_local_mgxs=args.rerun_local_mgxs,
        local_particles=args.local_particles,
        local_batches=args.local_batches,
        local_inactive=args.local_inactive,
        local_legendre_order=args.local_legendre_order,
        openmc_to_foam_particles=args.openmc_to_foam_particles,
        openmc_to_foam_batches=args.openmc_to_foam_batches,
        openmc_to_foam_inactive=args.openmc_to_foam_inactive,
        openmc_to_foam_legendre_order=args.openmc_to_foam_legendre_order,
        focus_fields=tuple(args.focus_fields),
        focus_zones=tuple(args.focus_zones) if args.focus_zones else None,
        focus_limit=args.focus_limit,
        threads=args.threads,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
