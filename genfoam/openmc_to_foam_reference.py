from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import openmc

from prepare_concentric_case import (
    DEFAULT_MGXS_EXPORT_DIR,
    DEFAULT_OUTPUT_DIR,
    _beta_by_precursor_group,
    _delayed_chi,
    _lambda_by_precursor_group,
    _prompt_chi,
    _reference_lambda_values,
    _sanitize_lambda_values,
    _sigma_pow,
    _sigma_removal,
    build_case_spec,
)


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "openmc_to_foam_reference"
CM_TO_M = 1.0e-2
EV_TO_J = 1.602176487e-19


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_statepoint(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("statepoint.*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No statepoint.*.h5 file found in {run_dir}")
    return candidates[-1]


def _resolve_tool_root(tool_root: Path | None) -> Path | None:
    if tool_root is not None:
        return tool_root
    configured = os.environ.get("OPENMCTOFOAM_ROOT")
    if configured:
        return Path(configured)
    return None


def _load_openmc_to_foam(tool_root: Path | None) -> tuple[Any, Any]:
    resolved_tool_root = _resolve_tool_root(tool_root)
    if resolved_tool_root is not None:
        if not resolved_tool_root.exists():
            raise FileNotFoundError(f"Could not find openmcToFoam tool root: {resolved_tool_root}")
        if not (resolved_tool_root / "MultiGroupXS").exists():
            raise FileNotFoundError(f"Could not find MultiGroupXS package under {resolved_tool_root}")
        sys.path.insert(0, str(resolved_tool_root))

    try:
        from MultiGroupXS import MultiGroupXS, MultiGroupXSManager  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import openmcToFoam's MultiGroupXS package. "
            "Install it in the active Python environment, set OPENMCTOFOAM_ROOT, "
            "or pass --tool-root /path/to/openmcToFoam."
        ) from exc

    return MultiGroupXS, MultiGroupXSManager


def _current_case_vectors(case_spec: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    precursor_group_count = len(case_spec["config"]["delayed_groups"])
    reference_lambda_values = _reference_lambda_values(case_spec, precursor_group_count)
    vectors: dict[str, dict[str, list[float]]] = {}
    for zone_name, material_payload in case_spec["materials"].items():
        group_constants = material_payload["group_constants_si"]
        group_count = len(group_constants["diffusion-coefficient"]["mean"])
        vectors[zone_name] = {
            "D": [float(value) for value in group_constants["diffusion-coefficient"]["mean"]],
            "nuSigmaEff": [float(value) for value in group_constants["nu-fission"]["mean"]],
            "sigmaPow": [float(value) for value in _sigma_pow(material_payload)],
            "sigmaRemoval": [float(value) for value in _sigma_removal(material_payload, group_count)],
            "chiPrompt": [float(value) for value in _prompt_chi(material_payload)],
            "chiDelayed": [float(value) for value in _delayed_chi(material_payload, group_count)],
            "Beta": [float(value) for value in _beta_by_precursor_group(material_payload, precursor_group_count)],
            "lambda": [
                float(value)
                for value in _sanitize_lambda_values(
                    _lambda_by_precursor_group(material_payload, precursor_group_count),
                    reference_lambda_values,
                )
            ],
        }
    return vectors


def _scale_openmc_to_foam_vectors(mgxs_cell: Any, master_flux: Any) -> dict[str, list[float]]:
    scattering_p0 = [
        np.transpose(moment_matrix)
        for moment_matrix in np.transpose(mgxs_cell.scatterMatrixXS.get_xs())
    ][0]
    sigma_removal = [
        total - scattering_p0[group_index][group_index]
        for group_index, total in enumerate(mgxs_cell.totalXS.get_xs())
    ]
    normalized_flux = [
        flux[0][0] / flux_master[0][0]
        for flux, flux_master in zip(mgxs_cell.integralFlux.mean, master_flux.mean, strict=True)
    ][::-1]
    return {
        "D": [float(value) * CM_TO_M for value in mgxs_cell.diffusionCoefficient.get_xs()],
        "nuSigmaEff": [float(value) / CM_TO_M for value in mgxs_cell.nuFissionXS.get_xs()],
        "sigmaPow": [float(value) * EV_TO_J / CM_TO_M for value in mgxs_cell.kappaFissionXS.get_xs()],
        "sigmaRemoval": [float(value) / CM_TO_M for value in sigma_removal],
        "chiPrompt": [float(value) for value in mgxs_cell.chiPrompt.get_xs()],
        "chiDelayed": [float(value) for value in mgxs_cell.chiDelayed.get_xs()[0]],
        "Beta": [float(value[0]) for value in mgxs_cell.beta.get_xs()],
        "lambda": [float(value) for value in np.asarray(mgxs_cell.Lambda.get_xs(), dtype=float).reshape(-1)],
        "integralFlux": [float(value) for value in normalized_flux],
    }


def _compare_vectors(reference: list[float], current: list[float]) -> dict[str, Any]:
    if len(reference) != len(current):
        return {
            "length_mismatch": {
                "reference": len(reference),
                "current": len(current),
            }
        }
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    abs_diff = np.abs(ref - cur)
    denom = np.maximum(np.abs(ref), 1.0e-16)
    rel_diff = abs_diff / denom
    return {
        "max_abs_diff": float(abs_diff.max(initial=0.0)),
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "max_rel_diff": float(rel_diff.max(initial=0.0)),
        "reference": reference,
        "current": current,
    }


def _build_reference_manager(
    model: openmc.Model,
    export: dict[str, Any],
    MultiGroupXS: Any,
    MultiGroupXSManager: Any,
) -> tuple[Any, dict[str, openmc.Cell]]:
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
    domains: dict[str, openmc.Cell] = {}
    for label, cell_name in export["config"]["domain_definitions"]:
        if cell_name not in cells_by_name:
            raise ValueError(f"Could not find cell {cell_name!r} for region {label!r} in the OpenMC model")
        fuel_fraction = 1.0 if label.startswith("core_fuel_ring_") else 0.0
        domain = cells_by_name[cell_name]
        domains[label] = domain
        manager.AddDomain(
            MultiGroupXS(
                domain,
                energy_groups,
                delayed_groups,
                fuelFraction=fuel_fraction,
                genfoamName=label,
            )
        )
    tallies = openmc.Tallies()
    manager.AddToTallies(tallies)
    model.tallies = tallies
    return manager, domains


def _run_reference_case(
    mgxs_export_dir: Path,
    output_dir: Path,
    tool_root: Path,
    particles: int,
    batches: int,
    inactive: int,
    threads: int,
) -> dict[str, Any]:
    export = _load_json(mgxs_export_dir / "outputs" / "mgxs_constants.json")
    model_xml_path = mgxs_export_dir / "reactor_run" / "model.xml"
    if not model_xml_path.exists():
        raise FileNotFoundError(f"Could not find OpenMC model XML at {model_xml_path}")

    MultiGroupXS, MultiGroupXSManager = _load_openmc_to_foam(tool_root)
    openmc.reset_auto_ids()
    model = openmc.Model.from_model_xml(model_xml_path)
    settings = model.settings
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    model.settings = settings

    manager, _domains = _build_reference_manager(model, export, MultiGroupXS, MultiGroupXSManager)

    run_dir = output_dir / "reactor_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        statepoint_path = Path(model.run(output=False, threads=threads)).resolve()
    finally:
        os.chdir(previous_cwd)

    nuclear_data_path = output_dir / "nuclearData.openmcToFoam"
    with openmc.StatePoint(statepoint_path) as statepoint:
        manager.LoadFromStatepoint(statepoint)
        manager.PrintForGeNFoam(str(nuclear_data_path))
        keff = {"mean": float(statepoint.keff.n), "std_dev": float(statepoint.keff.s)}

    master_flux = manager.masterMGXS.integralFlux
    reference_vectors = {
        mgxs_cell.genfoamName: _scale_openmc_to_foam_vectors(mgxs_cell, master_flux)
        for mgxs_cell in manager.mgxsList[2:]
    }
    return {
        "reference_vectors": reference_vectors,
        "statepoint_path": str(statepoint_path),
        "nuclear_data_path": str(nuclear_data_path),
        "keff": keff,
        "particles": particles,
        "batches": batches,
        "inactive": inactive,
        "threads": threads,
    }


def run_comparison(
    mgxs_export_dir: Path,
    output_dir: Path,
    tool_root: Path | None,
    particles: int,
    batches: int,
    inactive: int,
    threads: int,
) -> dict[str, Any]:
    case_spec = build_case_spec(mgxs_export_dir=mgxs_export_dir, output_dir=DEFAULT_OUTPUT_DIR)
    current_vectors = _current_case_vectors(case_spec)
    reference_result = _run_reference_case(
        mgxs_export_dir=mgxs_export_dir,
        output_dir=output_dir,
        tool_root=tool_root,
        particles=particles,
        batches=batches,
        inactive=inactive,
        threads=threads,
    )

    reference_vectors = reference_result["reference_vectors"]
    comparison: dict[str, Any] = {
        "zones": {},
        "notes": [
            "The current hand-written GeN-Foam generator writes integralFlux as unity, so integralFlux is not compared here.",
            "This reference path reruns OpenMC with openmcToFoam tallies; it does not reuse the existing MGXS export statepoints.",
        ],
    }
    for zone_name, zone_reference in reference_vectors.items():
        if zone_name not in current_vectors:
            comparison["zones"][zone_name] = {"missing_in_current_generator": True}
            continue
        zone_current = current_vectors[zone_name]
        comparison["zones"][zone_name] = {
            field_name: _compare_vectors(zone_reference[field_name], zone_current[field_name])
            for field_name in ("D", "nuSigmaEff", "sigmaPow", "sigmaRemoval", "chiPrompt", "chiDelayed", "Beta", "lambda")
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_vectors_path = output_dir / "reference_vectors.json"
    comparison_path = output_dir / "comparison.json"
    summary_path = output_dir / "summary.json"
    reference_vectors_path.write_text(json.dumps(reference_vectors, indent=2) + "\n", encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    summary = {
        "mgxs_export_dir": str(mgxs_export_dir),
        "tool_root": str(_resolve_tool_root(tool_root)) if _resolve_tool_root(tool_root) is not None else None,
        "reference_run": {key: value for key, value in reference_result.items() if key != "reference_vectors"},
        "reference_vectors_path": str(reference_vectors_path),
        "comparison_path": str(comparison_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an openmcToFoam reference OpenMC calculation and compare it with the local GeN-Foam nuclearData generator."
    )
    parser.add_argument("--mgxs-export-dir", type=Path, default=DEFAULT_MGXS_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REFERENCE_OUTPUT_DIR)
    parser.add_argument("--tool-root", type=Path, default=None)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--inactive", type=int, default=4)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_comparison(
        mgxs_export_dir=args.mgxs_export_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        tool_root=args.tool_root.resolve() if args.tool_root is not None else None,
        particles=args.particles,
        batches=args.batches,
        inactive=args.inactive,
        threads=args.threads,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
