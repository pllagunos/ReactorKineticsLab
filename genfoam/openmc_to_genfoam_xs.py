from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


CASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CASE_DIR.parent
OPENMC_DIR = PROJECT_DIR / "openmc"
if str(OPENMC_DIR) not in sys.path:
    sys.path.insert(0, str(OPENMC_DIR))

from mgxs_export import MGXSExportConfig, run_mgxs_export  # noqa: E402


CM_TO_M = 1.0e-2
EV_TO_J = 1.602176487e-19
DEFAULT_OUTPUT_SUBDIR = "mgxs_to_genfoam_xs"
DEFAULT_INACTIVE_FALLBACK = 1.0e-6
DEFAULT_MAX_DIFFUSION_M = 1.0e4
DEFAULT_PROMPT_GENERATION_TIME_S = 1.0
SCATTER_MATRIX_PRIORITY = (
    "consistent nu-scatter matrix",
    "nu-scatter matrix",
    "scatter matrix",
)


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


def _vector_entry(name: str, values: list[float]) -> str:
    return (
        f"                {name:<15} nonuniform List<scalar> {len(values)} "
        f"({' '.join(_format_number(value) for value in values)});"
    )


def _matrix_lines(name: str, matrix: list[list[float]]) -> list[str]:
    lines = [
        f"                {name} {len(matrix)} {len(matrix)}",
        "                (",
    ]
    for row in matrix:
        row_text = " ".join(_format_number(float(value)) for value in row)
        lines.append(f"                    ( {row_text} )")
    lines.append("                );")
    return lines


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


def select_scatter_matrix_payload(group_constants: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for xs_type in SCATTER_MATRIX_PRIORITY:
        payload = group_constants.get(xs_type)
        if payload is not None:
            return xs_type, payload
    available = ", ".join(sorted(group_constants))
    raise KeyError(
        "Could not find any supported scatter-matrix payload. "
        f"Expected one of {SCATTER_MATRIX_PRIORITY}, available: {available}"
    )


def normalize_mgxs_types_for_genfoam(mgxs_types: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = list(mgxs_types)
    if "chi-prompt" not in normalized:
        insert_at = normalized.index("chi") + 1 if "chi" in normalized else len(normalized)
        normalized.insert(insert_at, "chi-prompt")
    if "consistent nu-scatter matrix" not in normalized:
        insert_at = (
            normalized.index("nu-scatter matrix") + 1
            if "nu-scatter matrix" in normalized
            else len(normalized)
        )
        normalized.insert(insert_at, "consistent nu-scatter matrix")
    return tuple(normalized)


def validate_genfoam_scattering_contract(export: dict[str, Any]) -> None:
    config = export.get("config", {})
    if "scatter_correction" not in config or config["scatter_correction"] is not None:
        raise ValueError(
            "The MGXS export does not explicitly disable OpenMC's P0 scatter correction. "
            "Re-run prepare_concentric_case.py with --rerun-mgxs to generate GeN-Foam-compatible scattering data."
        )
    if config.get("scatter_formulation") != "consistent":
        raise ValueError(
            "The MGXS export does not declare the consistent scattering formulation. "
            "Re-run prepare_concentric_case.py with --rerun-mgxs."
        )
    if config.get("kinetics_group_count") != 1:
        raise ValueError(
            "The MGXS export does not use one-group Beta and decay-rate kinetics data. "
            "Re-run prepare_concentric_case.py with --rerun-mgxs."
        )

    missing_domains = [
        zone_name
        for zone_name, domain_payload in export.get("domains", {}).items()
        if (
            "consistent nu-scatter matrix" not in domain_payload.get("group_constants", {})
            or "chi-prompt" not in domain_payload.get("group_constants", {})
        )
    ]
    if missing_domains:
        raise ValueError(
            "The MGXS export is missing GeN-Foam prompt/scattering data for: "
            f"{', '.join(missing_domains)}. Re-run prepare_concentric_case.py with --rerun-mgxs."
        )


def _resolve_prompt_generation_time(run_payload: dict[str, Any]) -> tuple[float, str]:
    prompt = run_payload.get("prompt_generation_time_s")
    if isinstance(prompt, (int, float)) and float(prompt) > 0.0:
        return float(prompt), "export"

    generation = run_payload.get("generation_time_s")
    if isinstance(generation, dict):
        mean = generation.get("mean")
        if isinstance(mean, (int, float)) and float(mean) > 0.0:
            return float(mean), "export_generation_time"
    elif isinstance(generation, (int, float)) and float(generation) > 0.0:
        return float(generation), "export_generation_time"

    return DEFAULT_PROMPT_GENERATION_TIME_S, "fallback_constant"


def _normalize_energy_distribution(values: list[float]) -> list[float]:
    total = sum(float(value) for value in values)
    if total <= 0.0:
        return [0.0] * len(values)
    return [float(value) / total for value in values]


def _collapse_beta_by_delayed_group(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    if not values:
        return []
    if isinstance(values[0], list):
        return [float(row[0]) if row else 0.0 for row in values]
    return [float(value) for value in values]


def _extract_source_export(
    mgxs_export_dir: Path,
    output_dir: Path,
    rerun_mgxs: bool,
    particles: int | None,
    batches: int | None,
    inactive: int | None,
    threads: int | None,
    legendre_order: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mgxs_json_path = mgxs_export_dir / "outputs" / "mgxs_constants.json"
    export = _load_json(mgxs_json_path)
    model_xml_path = mgxs_export_dir / "reactor_run" / "model.xml"

    if not rerun_mgxs:
        validate_genfoam_scattering_contract(export)
        return export, {
            "mode": "existing_export",
            "mgxs_json_path": str(mgxs_json_path.resolve()),
            "model_xml_path": str(model_xml_path.resolve()),
        }

    config_payload = export["config"]
    configured_run = export.get("model", {}).get("configured_run", {})
    normalized_mgxs_types = normalize_mgxs_types_for_genfoam(tuple(str(item) for item in config_payload["mgxs_types"]))
    config = MGXSExportConfig(
        particles=particles if particles is not None else int(configured_run.get("particles", 16000)),
        batches=batches if batches is not None else int(configured_run.get("batches", 20)),
        inactive=inactive if inactive is not None else int(configured_run.get("inactive", 5)),
        domain_definitions=tuple(tuple(item) for item in config_payload["domain_definitions"]),
        energy_group_edges_ev=tuple(float(edge) for edge in config_payload["energy_group_edges_ev"]),
        mgxs_types=normalized_mgxs_types,
        delayed_groups=tuple(int(item) for item in config_payload["delayed_groups"]),
        legendre_order=(
            int(legendre_order)
            if legendre_order is not None
            else int(configured_run.get("legendre_order", config_payload["legendre_order"]))
        ),
        scatter_correction=None,
        scatter_formulation="consistent",
        kinetics_group_count=1,
    )
    rerun_export_dir = output_dir / "mgxs_rerun_export"
    rerun_results = run_mgxs_export(
        model_path_or_dir=model_xml_path,
        config=config,
        export_dir=rerun_export_dir,
        threads=threads,
    )
    validate_genfoam_scattering_contract(rerun_results)
    return rerun_results, {
        "mode": "rerun_export",
        "mgxs_json_path": str((rerun_export_dir / "outputs" / "mgxs_constants.json").resolve()),
        "model_xml_path": str(model_xml_path.resolve()),
        "rerun_export_dir": str(rerun_export_dir.resolve()),
    }


def _raw_zone_data_from_export(zone_name: str, export: dict[str, Any]) -> dict[str, Any]:
    domain_payload = export["domains"][zone_name]
    group_constants = domain_payload["group_constants"]
    delayed = domain_payload.get("delayed_neutrons", {})
    genfoam_aux = domain_payload.get("genfoam_aux", {})
    scatter_matrix_source, scatter_matrix_payload = select_scatter_matrix_payload(group_constants)

    total_xs = [float(value) * 100.0 for value in group_constants["total"]["mean"]]
    diffusion = [float(value) * CM_TO_M for value in group_constants["diffusion-coefficient"]["mean"]]
    inverse_velocity = [float(value) / CM_TO_M for value in group_constants["inverse-velocity"]["mean"]]
    nu_sigma_eff = [float(value) / CM_TO_M for value in group_constants["nu-fission"]["mean"]]
    sigma_pow = [float(value) * EV_TO_J / CM_TO_M for value in group_constants["kappa-fission"]["mean"]]
    chi_prompt = [float(value) for value in group_constants["chi-prompt"]["mean"]]

    scattering_p0 = [
        [float(value) * 100.0 for value in row]
        for row in _collapse_legendre_moment(scatter_matrix_payload["mean"])
    ]
    sigma_removal = [
        total - scattering_p0[group_index][group_index]
        for group_index, total in enumerate(total_xs)
    ]

    beta_by_delayed_group = _collapse_beta_by_delayed_group(delayed.get("beta_total_by_delayed_group", []))
    if not beta_by_delayed_group:
        beta_by_delayed_group = _collapse_beta_by_delayed_group(delayed.get("beta_by_delayed_group", []))
    lambda_values = [float(value) for value in delayed.get("decay_rate_per_s_by_delayed_group", [])]

    synthesized_fields: list[str] = []
    chi_delayed_payload = genfoam_aux.get("chi_delayed", {})
    chi_delayed = chi_delayed_payload.get("mean")
    if chi_delayed is None:
        chi_delayed = _normalize_energy_distribution(
            [float(value) for value in delayed.get("beta_total_by_energy_group", [0.0] * len(total_xs))]
        )
        synthesized_fields.append("chiDelayed")
    else:
        chi_delayed = [float(value) for value in chi_delayed]

    integral_flux = genfoam_aux.get("integral_flux")
    if integral_flux is None:
        integral_flux = [1.0] * len(total_xs)
        synthesized_fields.append("integralFlux")
    else:
        integral_flux = [float(value) for value in integral_flux]

    return {
        "cell_name": str(domain_payload["domain"]["name"]),
        "fuelFraction": 1.0 if zone_name.startswith("core_fuel_ring_") else 0.0,
        "IV": inverse_velocity,
        "D": diffusion,
        "nuSigmaEff": nu_sigma_eff,
        "sigmaPow": sigma_pow,
        "scatteringMatrixP0": scattering_p0,
        "sigmaRemoval": sigma_removal,
        "chiPrompt": chi_prompt,
        "chiDelayed": chi_delayed,
        "Beta": beta_by_delayed_group,
        "lambda": lambda_values,
        "discFactor": [1.0] * len(total_xs),
        "integralFlux": integral_flux,
        "metadata": {
            "synthesized_fields": synthesized_fields,
            "source_domain_id": domain_payload["domain"]["id"],
            "scatter_matrix_source": scatter_matrix_source,
        },
    }


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
                f"MGXS export produced an invalid {field_name} value for active group {group_index + 1} in {zone_name}"
            )
        sanitized.append(fallback)
    return sanitized


def _sanitize_sigma_removal(zone_name: str, values: list[float]) -> list[float]:
    positive_values = [value for value in values if value > 0.0 and math.isfinite(value)]
    replacement = min(positive_values) if positive_values else DEFAULT_INACTIVE_FALLBACK
    sanitized: list[float] = []
    for value in values:
        if value > 0.0 and math.isfinite(value):
            sanitized.append(value)
            continue
        # Low-stat MGXS reruns can produce slightly negative removal in active
        # groups from tally noise. GeN-Foam still needs a positive removal term.
        sanitized.append(replacement)
    return sanitized


def _sanitize_diffusion(zone_name: str, zone_data: dict[str, Any]) -> list[float]:
    sanitized = [float(value) for value in zone_data["D"]]
    for group_index, value in enumerate(sanitized):
        needs_fix = not math.isfinite(value) or value <= 0.0 or value > DEFAULT_MAX_DIFFUSION_M
        if not needs_fix:
            continue
        if not _is_inactive_group(zone_data, group_index):
            raise ValueError(
                f"MGXS export produced an invalid diffusion coefficient in active group {group_index + 1} for {zone_name}"
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


def _sanitize_lambda(values: list[float]) -> list[float]:
    sanitized: list[float] = []
    for value in values:
        if value > 0.0 and math.isfinite(value):
            sanitized.append(value)
        else:
            sanitized.append(DEFAULT_INACTIVE_FALLBACK)
    return sanitized


def _changed_indices(reference: list[float], current: list[float], tolerance: float = 1.0e-14) -> list[int]:
    return [
        index + 1
        for index, (raw_value, adapted_value) in enumerate(zip(reference, current, strict=True))
        if abs(float(raw_value) - float(adapted_value)) > tolerance
    ]


def _adapt_zone_data(zone_name: str, raw_zone_data: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(raw_zone_data)
    adapted["IV"] = _sanitize_positive_vector(zone_name, "IV", raw_zone_data["IV"], raw_zone_data)
    adapted["D"] = _sanitize_diffusion(zone_name, raw_zone_data)
    adapted["sigmaRemoval"] = _sanitize_sigma_removal(zone_name, raw_zone_data["sigmaRemoval"])
    adapted["lambda"] = _sanitize_lambda(raw_zone_data["lambda"])
    adapted["integralFlux"] = [
        float(value) if math.isfinite(value) and value >= 0.0 else 0.0
        for value in raw_zone_data["integralFlux"]
    ]
    adapted["metadata"] = {
        **raw_zone_data.get("metadata", {}),
        "sanitization": {
            "IV_groups": _changed_indices(raw_zone_data["IV"], adapted["IV"]),
            "D_groups": _changed_indices(raw_zone_data["D"], adapted["D"]),
            "sigmaRemoval_groups": _changed_indices(raw_zone_data["sigmaRemoval"], adapted["sigmaRemoval"]),
            "lambda_groups": _changed_indices(raw_zone_data["lambda"], adapted["lambda"]),
            "integralFlux_groups": _changed_indices(raw_zone_data["integralFlux"], adapted["integralFlux"]),
        },
    }
    return adapted


def build_nuclear_data_text_from_export(
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
        description = (
            f"                // MGXS-derived XS mapped to {zone_block_count} structured Gmsh mesh cell(s)."
            if zone_block_count
            else "                // MGXS-derived XS."
        )
        lines.extend([
            f"            {zone_name}",
            "            {",
            description,
            f"                fuelFraction    {_format_number(zone_data['fuelFraction'])};",
            _vector_entry("sigmaRemoval", zone_data["sigmaRemoval"]),
            _vector_entry("nuSigmaEff", zone_data["nuSigmaEff"]),
            _vector_entry("sigmaPow", zone_data["sigmaPow"]),
        ])
        lines.extend(_matrix_lines("scatteringMatrixP0", zone_data["scatteringMatrixP0"]))
        lines.extend([
            _vector_entry("discFactor", zone_data["discFactor"]),
            _vector_entry("chiPrompt", zone_data["chiPrompt"]),
            _vector_entry("chiDelayed", zone_data["chiDelayed"]),
            _vector_entry("IV", zone_data["IV"]),
            _vector_entry("D", zone_data["D"]),
            _vector_entry("integralFlux", zone_data["integralFlux"]),
            _vector_entry("lambda", zone_data["lambda"]),
            _vector_entry("Beta", zone_data["Beta"]),
            "            }",
        ])

    lines.extend([
        "        );",
        "    }",
        ");",
        "",
    ])
    return "\n".join(lines) + "\n"


def generate_genfoam_xs(
    mgxs_export_dir: Path,
    output_dir: Path,
    rerun_mgxs: bool = False,
    particles: int | None = None,
    batches: int | None = None,
    inactive: int | None = None,
    threads: int | None = None,
    legendre_order: int | None = None,
) -> dict[str, Any]:
    mgxs_export_dir = mgxs_export_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    export, source_metadata = _extract_source_export(
        mgxs_export_dir=mgxs_export_dir,
        output_dir=output_dir,
        rerun_mgxs=rerun_mgxs,
        particles=particles,
        batches=batches,
        inactive=inactive,
        threads=threads,
        legendre_order=legendre_order,
    )

    domain_order = [label for label, _ in export["config"]["domain_definitions"]]
    prompt_generation_time_s, prompt_generation_time_source = _resolve_prompt_generation_time(export.get("run", {}))

    raw_zones = {
        zone_name: _raw_zone_data_from_export(zone_name, export)
        for zone_name in domain_order
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
            **source_metadata,
        },
        "config": {
            "group_count": len(export["config"]["energy_group_edges_ev"]) - 1,
            "energy_group_edges_ev": export["config"]["energy_group_edges_ev"],
            "precursor_group_count": len(export["config"]["delayed_groups"]),
            "delayed_groups": export["config"]["delayed_groups"],
            "domain_order": domain_order,
            "legendre_order": int(export["config"]["legendre_order"]),
            "scatter_correction": export["config"]["scatter_correction"],
            "scatter_formulation": export["config"]["scatter_formulation"],
            "kinetics_group_count": int(export["config"]["kinetics_group_count"]),
            "scatter_matrix_sources": {
                zone_name: raw_zones[zone_name]["metadata"]["scatter_matrix_source"]
                for zone_name in domain_order
            },
        },
        "reference_run": {
            "keff": export.get("run", {}).get("keff"),
            "prompt_generation_time_s": prompt_generation_time_s,
            "prompt_generation_time_source": prompt_generation_time_source,
            "run": export.get("run", {}),
            "configured_run": export.get("model", {}).get("configured_run", {}),
        },
        "raw_zones": raw_zones,
        "zones": adapted_zones,
        "notes": [
            "Reference vectors are the MGXS-export-derived zone values.",
            "Current vectors are the sanitized GeN-Foam writer outputs.",
            "Non-zero deltas should be interpreted as writer interventions or legacy-export fallbacks.",
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
            "adapted_nuclear_data": str(adapted_nuclear_data_path),
        },
    }
    _write_json(summary_path, summary)

    payload["files"] = {
        "summary": str(summary_path),
        "raw_vectors": str(raw_vectors_path),
        "adapted_vectors": str(adapted_vectors_path),
        "adapted_nuclear_data": str(adapted_nuclear_data_path),
    }
    return payload
