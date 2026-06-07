from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import openmc

# where is the 'from MultiGroupXS import *' ?


CM_TO_M = 1.0e-2
EV_TO_J = 1.602176487e-19
DEFAULT_OUTPUT_SUBDIR = "openmc_to_foam_xs"
DEFAULT_INACTIVE_FALLBACK = 1.0e-6
DEFAULT_MAX_DIFFUSION_M = 1.0e4


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def resolve_tool_root(tool_root: Path | None) -> Path | None:
    if tool_root is not None:
        return tool_root
    configured = os.environ.get("OPENMCTOFOAM_ROOT")
    if configured:
        return Path(configured)
    return None


def _load_openmc_to_foam(tool_root: Path | None) -> tuple[Any, Any]:
    resolved_tool_root = resolve_tool_root(tool_root)
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
            "or pass --openmc-to-foam-tool-root /path/to/openmcToFoam."
        ) from exc

    return MultiGroupXS, MultiGroupXSManager


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


def _matrix_p0(scatter_xs: Any) -> list[list[float]]:
    p0 = [np.transpose(moment_matrix) for moment_matrix in np.transpose(scatter_xs)][0]
    return [[float(value) * 100.0 for value in row] for row in p0]


def _normalize_integral_flux(integral_flux: Any, master_flux: Any) -> list[float]:
    normalized: list[float] = []
    for flux, flux_master in zip(integral_flux.mean, master_flux.mean, strict=True):
        denominator = float(flux_master[0][0])
        numerator = float(flux[0][0])
        normalized.append(0.0 if denominator == 0.0 else numerator / denominator)
    return normalized[::-1]


def _is_inactive_group(zone_data: dict[str, Any], group_index: int) -> bool:
    scatter = zone_data["scatteringMatrixP0"]
    scatter_row = scatter[group_index]
    scatter_col = [row[group_index] for row in scatter]
    return (
        zone_data["nuSigmaEff"][group_index] == 0.0
        and zone_data["sigmaPow"][group_index] == 0.0
        and zone_data["chiPrompt"][group_index] == 0.0
        and zone_data["chiDelayed"][group_index] == 0.0
        and all(value == 0.0 for value in scatter_row)
        and all(value == 0.0 for value in scatter_col)
    )


def _sanitize_positive_vector(
    zone_name: str,
    field_name: str,
    values: list[float],
    zone_data: dict[str, Any],
    fallback: float = DEFAULT_INACTIVE_FALLBACK,
) -> list[float]:
    sanitized: list[float] = []
    for group_index, value in enumerate(values):
        if value > 0.0 and math.isfinite(value):
            sanitized.append(value)
            continue
        if not _is_inactive_group(zone_data, group_index):
            raise ValueError(
                f"openmcToFoam produced an invalid {field_name} value for active group {group_index + 1} in {zone_name}"
            )
        sanitized.append(fallback)
    return sanitized


def _sanitize_diffusion(zone_name: str, zone_data: dict[str, Any]) -> list[float]:
    sanitized = [float(value) for value in zone_data["D"]]
    for group_index, value in enumerate(sanitized):
        needs_fix = not math.isfinite(value) or value <= 0.0 or value > DEFAULT_MAX_DIFFUSION_M
        if not needs_fix:
            continue
        if not _is_inactive_group(zone_data, group_index):
            raise ValueError(
                f"openmcToFoam produced an invalid diffusion coefficient in active group {group_index + 1} for {zone_name}"
            )
        replacement = next(
            (
                candidate
                for other_index, candidate in enumerate(sanitized)
                if other_index != group_index and math.isfinite(candidate) and 0.0 < candidate <= DEFAULT_MAX_DIFFUSION_M
            ),
            1.0,
        )
        sanitized[group_index] = replacement
    return sanitized


def _sanitize_lambda(values: list[float], fallback_values: list[float]) -> list[float]:
    sanitized: list[float] = []
    for index, value in enumerate(values):
        if value > 0.0 and math.isfinite(value):
            sanitized.append(value)
            continue
        fallback = fallback_values[index] if index < len(fallback_values) else DEFAULT_INACTIVE_FALLBACK
        sanitized.append(fallback if fallback > 0.0 and math.isfinite(fallback) else DEFAULT_INACTIVE_FALLBACK)
    return sanitized


def _extract_raw_zone_data(mgxs_cell: Any, master_flux: Any, master_lambda: list[float]) -> dict[str, Any]:
    scattering_matrix_p0 = _matrix_p0(mgxs_cell.scatterMatrixXS.get_xs())
    sigma_removal = [
        float(total - scattering_matrix_p0[group_index][group_index] / 100.0) / CM_TO_M
        for group_index, total in enumerate(mgxs_cell.totalXS.get_xs())
    ]
    lambda_values = [float(value) for value in np.asarray(mgxs_cell.Lambda.get_xs(), dtype=float).reshape(-1)]
    if not any(value > 0.0 and math.isfinite(value) for value in lambda_values):
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


def _adapt_zone_data(zone_name: str, raw_zone_data: dict[str, Any], fallback_lambda: list[float]) -> dict[str, Any]:
    adapted = dict(raw_zone_data)
    adapted["IV"] = _sanitize_positive_vector(zone_name, "IV", raw_zone_data["IV"], raw_zone_data)
    adapted["D"] = _sanitize_diffusion(zone_name, raw_zone_data)
    adapted["sigmaRemoval"] = _sanitize_positive_vector(
        zone_name,
        "sigmaRemoval",
        raw_zone_data["sigmaRemoval"],
        raw_zone_data,
    )
    adapted["lambda"] = _sanitize_lambda(raw_zone_data["lambda"], fallback_lambda)
    adapted["integralFlux"] = [
        float(value) if math.isfinite(value) and value >= 0.0 else 0.0
        for value in raw_zone_data["integralFlux"]
    ]
    return adapted


def build_nuclear_data_text_from_openmc_to_foam(
    xs_payload: dict[str, Any],
    domain_order: list[str],
    zone_counts: dict[str, int] | None = None,
) -> str:
    zone_counts = zone_counts or {}
    group_count = int(xs_payload["config"]["group_count"])
    precursor_group_count = int(xs_payload["config"]["precursor_group_count"])
    lines = [
        _dictionary_header("nuclearData").rstrip(),
        f"promptGenerationTime {_format_number(xs_payload['reference_run']['prompt_generation_time_s'])};",
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

    for zone_name in domain_order:
        zone_data = xs_payload["zones"].get(zone_name)
        if zone_data is None:
            continue
        zone_block_count = zone_counts.get(zone_name, 0)
        lines.extend([
            f"            {zone_name}",
            "            {",
            (
                f"                // openmcToFoam-derived XS mapped to {zone_block_count} structured Gmsh mesh cell(s)."
                if zone_block_count
                else "                // openmcToFoam-derived XS."
            ),
            f"                fuelFraction    {_format_number(zone_data['fuelFraction'])};",
            f"                sigmaRemoval    nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['sigmaRemoval'])});",
            f"                nuSigmaEff      nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['nuSigmaEff'])});",
            f"                sigmaPow        nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['sigmaPow'])});",
            f"                scatteringMatrixP0 {group_count} {group_count}",
            "                (",
        ])
        for row in zone_data["scatteringMatrixP0"]:
            lines.append(f"                    ( {_format_scalar_list([float(value) for value in row])} )")
        lines.extend([
            "                );",
            f"                discFactor      nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['discFactor'])});",
            f"                chiPrompt       nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['chiPrompt'])});",
            f"                chiDelayed      nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['chiDelayed'])});",
            f"                IV              nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['IV'])});",
            f"                D               nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['D'])});",
            f"                integralFlux    nonuniform List<scalar> {group_count} ({_format_scalar_list(zone_data['integralFlux'])});",
            f"                lambda          nonuniform List<scalar> {precursor_group_count} ({_format_scalar_list(zone_data['lambda'])});",
            f"                Beta            nonuniform List<scalar> {precursor_group_count} ({_format_scalar_list(zone_data['Beta'])});",
            "            }",
        ])

    lines.extend([
        "        );",
        "    }",
        ");",
        "",
    ])
    return "\n".join(lines) + "\n"


def generate_openmc_to_foam_xs(
    mgxs_export_dir: Path,
    output_dir: Path,
    tool_root: Path | None = None,
    particles: int | None = None,
    batches: int | None = None,
    inactive: int | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    mgxs_export_dir = mgxs_export_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    export = _load_json(mgxs_export_dir / "outputs" / "mgxs_constants.json")
    model_xml_path = mgxs_export_dir / "reactor_run" / "model.xml"
    if not model_xml_path.exists():
        raise FileNotFoundError(f"Could not find OpenMC model XML at {model_xml_path}")

    MultiGroupXS, MultiGroupXSManager = _load_openmc_to_foam(tool_root)
    openmc.reset_auto_ids()
    model = openmc.Model.from_model_xml(model_xml_path)
    settings = model.settings
    settings.run_mode = "eigenvalue"
    if particles is not None:
        settings.particles = particles
    if batches is not None:
        settings.batches = batches
    if inactive is not None:
        settings.inactive = inactive
    model.settings = settings
    resolved_threads = threads if threads is not None else max(1, os.cpu_count() or 1)

    manager, _domains = _build_reference_manager(model, export, MultiGroupXS, MultiGroupXSManager)

    run_dir = output_dir / "reactor_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        statepoint_path = Path(model.run(output=False, threads=resolved_threads)).resolve()
    finally:
        os.chdir(previous_cwd)

    raw_nuclear_data_path = output_dir / "nuclearData.openmcToFoam"
    with openmc.StatePoint(statepoint_path) as statepoint:
        manager.LoadFromStatepoint(statepoint)
        os.chdir(output_dir)
        try:
            manager.PrintForGeNFoam(raw_nuclear_data_path.name)
        finally:
            os.chdir(previous_cwd)
        keff = {"mean": float(statepoint.keff.n), "std_dev": float(statepoint.keff.s)}
        prompt_generation_time_s = float(manager.masterMGXSOneGroup.GetPromptGenerationLifetime()[0])

    master_flux = manager.masterMGXS.integralFlux
    master_lambda = [float(value) for value in np.asarray(manager.masterMGXSOneGroup.Lambda.get_xs(), dtype=float).reshape(-1)]
    domain_order = [label for label, _ in export["config"]["domain_definitions"]]

    raw_zones: dict[str, dict[str, Any]] = {}
    for mgxs_cell in manager.mgxsList[2:]:
        raw_zones[mgxs_cell.genfoamName] = _extract_raw_zone_data(
            mgxs_cell=mgxs_cell,
            master_flux=master_flux,
            master_lambda=master_lambda,
        )

    adapted_zones = {
        zone_name: _adapt_zone_data(zone_name, raw_zone_data, master_lambda)
        for zone_name, raw_zone_data in raw_zones.items()
    }

    payload = {
        "source": {
            "mgxs_export_dir": str(mgxs_export_dir),
            "model_xml_path": str(model_xml_path.resolve()),
            "tool_root": str(resolve_tool_root(tool_root)) if resolve_tool_root(tool_root) is not None else None,
        },
        "config": {
            "group_count": len(export["config"]["energy_group_edges_ev"]) - 1,
            "energy_group_edges_ev": export["config"]["energy_group_edges_ev"],
            "precursor_group_count": len(export["config"]["delayed_groups"]),
            "delayed_groups": export["config"]["delayed_groups"],
            "domain_order": domain_order,
        },
        "reference_run": {
            "statepoint_path": str(statepoint_path),
            "raw_nuclear_data_path": str(raw_nuclear_data_path),
            "keff": keff,
            "prompt_generation_time_s": prompt_generation_time_s,
            "particles": int(model.settings.particles),
            "batches": int(model.settings.batches),
            "inactive": int(model.settings.inactive),
            "threads": int(resolved_threads),
        },
        "raw_zones": raw_zones,
        "zones": adapted_zones,
    }

    adapted_nuclear_data_text = build_nuclear_data_text_from_openmc_to_foam(payload, domain_order)
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
        "files": {
            "raw_vectors": str(raw_vectors_path),
            "adapted_vectors": str(adapted_vectors_path),
            "raw_nuclear_data": str(raw_nuclear_data_path),
            "adapted_nuclear_data": str(adapted_nuclear_data_path),
        },
    }
    _write_json(summary_path, summary)

    payload["files"] = summary["files"] | {"summary": str(summary_path)}
    return payload
