from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openmc

from ploting import resolve_openmc_exec


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_XML_PATH = SCRIPT_DIR / "build" / "concentric" / "model.xml"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "build" / "concentric" / "reactivity_coefficients"
)
DEFAULT_REFERENCE_RESULTS_DIR = (
    SCRIPT_DIR
    / "reference_data"
    / "concentric"
    / "reactivity_coefficients"
    / "results"
)

FAST_PARTICLES = 10_000
FAST_BATCHES = 40
FAST_INACTIVE = 10
PRODUCTION_PARTICLES = 200_000
PRODUCTION_BATCHES = 100
PRODUCTION_INACTIVE = 10

FUEL_MATERIAL_NAME = "U3Si2 fuel"
D2O_MATERIAL_NAME = "Heavy water moderator"
REQUIRED_MATERIALS = (FUEL_MATERIAL_NAME, D2O_MATERIAL_NAME)


PerturbationKind = Literal["temperature", "density"]


@dataclass(frozen=True)
class TransportSettings:
    particles: int
    batches: int
    inactive: int
    threads: int
    openmc_exec: str
    base_seed: int
    keep_tallies: bool


@dataclass(frozen=True)
class MaterialBaseline:
    name: str
    temperature_k: float
    density_g_per_cm3: float
    density_units: str


@dataclass(frozen=True)
class CoefficientDefinition:
    name: str
    label: str
    material_name: str
    kind: PerturbationKind
    baseline_value: float
    delta_value: float
    variable_unit: str
    coefficient_unit: str


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    coefficient_name: str
    case_name: str
    material_name: str | None
    kind: PerturbationKind | None
    baseline_value: float | None
    perturbed_value: float | None
    variable_unit: str | None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    coefficient_name: str
    case_name: str
    replicate: int
    seed: int
    material_name: str | None
    kind: PerturbationKind | None
    baseline_value: float | None
    perturbed_value: float | None
    variable_unit: str | None
    k_eff: float
    k_eff_std: float
    rho_pcm: float
    rho_std_pcm: float
    run_dir: str
    statepoint_path: str


@dataclass(frozen=True)
class AveragedCaseResult:
    case_id: str
    coefficient_name: str
    case_name: str
    material_name: str | None
    kind: PerturbationKind | None
    baseline_value: float | None
    perturbed_value: float | None
    variable_unit: str | None
    replicate_count: int
    k_eff: float
    k_eff_std: float
    rho_pcm: float
    rho_std_pcm: float


@dataclass(frozen=True)
class CoefficientResult:
    name: str
    label: str
    material_name: str
    kind: PerturbationKind
    baseline_value: float
    minus_value: float
    plus_value: float
    delta_value: float
    variable_unit: str
    coefficient_value: float
    coefficient_std: float
    coefficient_unit: str
    base_rho_pcm: float | None
    base_rho_std_pcm: float | None
    minus_rho_pcm: float
    minus_rho_std_pcm: float
    plus_rho_pcm: float
    plus_rho_std_pcm: float
    coefficient_per_percent_density: float | None = None
    coefficient_per_percent_density_std: float | None = None


def rho_pcm(k_eff: float) -> float:
    return (k_eff - 1.0) / k_eff * 1.0e5


def rho_std_pcm(k_eff: float, k_eff_std: float) -> float:
    return abs(k_eff_std / (k_eff * k_eff) * 1.0e5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_xml_path(model_path_or_dir: str | Path) -> Path:
    candidate = Path(model_path_or_dir).expanduser()
    model_xml_path = candidate if candidate.name == "model.xml" else candidate / "model.xml"
    if not model_xml_path.exists():
        raise FileNotFoundError(f"Could not find model.xml at {model_xml_path}")
    return model_xml_path.resolve()


def load_source_model(model_xml_path: Path) -> openmc.Model:
    openmc.reset_auto_ids()
    model = openmc.Model.from_model_xml(model_xml_path)
    if model.geometry is None:
        raise ValueError(f"Loaded model at {model_xml_path} does not contain geometry")
    if model.settings.source is None:
        raise ValueError(
            f"Loaded model at {model_xml_path} does not contain a source definition"
        )
    return model


def material_by_name(model: openmc.Model, material_name: str) -> openmc.Material:
    if model.materials is None:
        raise ValueError("Model does not contain a materials collection")

    matches = [material for material in model.materials if material.name == material_name]
    if not matches:
        raise ValueError(f"Required material {material_name!r} was not found")
    if len(matches) > 1:
        raise ValueError(f"Material name {material_name!r} is not unique")
    return matches[0]


def collect_material_baselines(model: openmc.Model) -> dict[str, MaterialBaseline]:
    baselines: dict[str, MaterialBaseline] = {}
    for name in REQUIRED_MATERIALS:
        material = material_by_name(model, name)
        if material.temperature is None:
            raise ValueError(f"Material {name!r} does not define a temperature")
        if material.density is None:
            raise ValueError(f"Material {name!r} does not define a density")
        if material.density_units != "g/cm3":
            raise ValueError(
                f"Material {name!r} density units are {material.density_units!r}; "
                "expected 'g/cm3'"
            )
        baselines[name] = MaterialBaseline(
            name=name,
            temperature_k=float(material.temperature),
            density_g_per_cm3=float(material.density),
            density_units=str(material.density_units),
        )
    return baselines


def build_coefficient_definitions(
    baselines: dict[str, MaterialBaseline],
    *,
    fuel_temperature_delta_k: float,
    moderator_temperature_delta_k: float,
    moderator_density_delta_fraction: float,
) -> list[CoefficientDefinition]:
    if fuel_temperature_delta_k <= 0.0:
        raise ValueError("fuel_temperature_delta_k must be positive")
    if moderator_temperature_delta_k <= 0.0:
        raise ValueError("moderator_temperature_delta_k must be positive")
    if moderator_density_delta_fraction <= 0.0:
        raise ValueError("moderator_density_delta_fraction must be positive")

    fuel = baselines[FUEL_MATERIAL_NAME]
    moderator = baselines[D2O_MATERIAL_NAME]
    density_delta = moderator.density_g_per_cm3 * moderator_density_delta_fraction

    if moderator.density_g_per_cm3 - density_delta <= 0.0:
        raise ValueError("D2O density perturbation would make density non-positive")

    return [
        CoefficientDefinition(
            name="fuel_temperature",
            label="Fuel temperature",
            material_name=FUEL_MATERIAL_NAME,
            kind="temperature",
            baseline_value=fuel.temperature_k,
            delta_value=fuel_temperature_delta_k,
            variable_unit="K",
            coefficient_unit="pcm/K",
        ),
        CoefficientDefinition(
            name="d2o_temperature",
            label="D2O moderator temperature",
            material_name=D2O_MATERIAL_NAME,
            kind="temperature",
            baseline_value=moderator.temperature_k,
            delta_value=moderator_temperature_delta_k,
            variable_unit="K",
            coefficient_unit="pcm/K",
        ),
        CoefficientDefinition(
            name="d2o_density",
            label="D2O moderator density",
            material_name=D2O_MATERIAL_NAME,
            kind="density",
            baseline_value=moderator.density_g_per_cm3,
            delta_value=density_delta,
            variable_unit="g/cm3",
            coefficient_unit="pcm/(g/cm3)",
        ),
    ]


def build_case_definitions(
    coefficient_definitions: list[CoefficientDefinition],
) -> list[CaseDefinition]:
    cases = [
        CaseDefinition(
            case_id="base",
            coefficient_name="base",
            case_name="base",
            material_name=None,
            kind=None,
            baseline_value=None,
            perturbed_value=None,
            variable_unit=None,
        )
    ]
    for definition in coefficient_definitions:
        cases.extend(
            [
                CaseDefinition(
                    case_id=f"{definition.name}_minus",
                    coefficient_name=definition.name,
                    case_name="minus",
                    material_name=definition.material_name,
                    kind=definition.kind,
                    baseline_value=definition.baseline_value,
                    perturbed_value=definition.baseline_value
                    - definition.delta_value,
                    variable_unit=definition.variable_unit,
                ),
                CaseDefinition(
                    case_id=f"{definition.name}_plus",
                    coefficient_name=definition.name,
                    case_name="plus",
                    material_name=definition.material_name,
                    kind=definition.kind,
                    baseline_value=definition.baseline_value,
                    perturbed_value=definition.baseline_value
                    + definition.delta_value,
                    variable_unit=definition.variable_unit,
                ),
            ]
        )
    return cases


def apply_case_to_model(model: openmc.Model, case: CaseDefinition) -> None:
    if case.material_name is None:
        return
    if case.kind is None or case.perturbed_value is None:
        raise ValueError(f"Case {case.case_id!r} is missing perturbation metadata")

    material = material_by_name(model, case.material_name)
    if case.kind == "temperature":
        material.temperature = float(case.perturbed_value)
    elif case.kind == "density":
        material.set_density("g/cm3", float(case.perturbed_value))
    else:
        raise ValueError(f"Unsupported perturbation kind {case.kind!r}")


def _is_mesh_tally(tally: openmc.Tally) -> bool:
    return any("Mesh" in type(tally_filter).__name__ for tally_filter in tally.filters)


def remove_mesh_tallies(model: openmc.Model) -> dict[str, int]:
    if model.tallies is None:
        return {"total": 0, "removed_mesh": 0, "retained_non_mesh": 0}

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


def configure_run_settings(
    model: openmc.Model,
    transport: TransportSettings,
    *,
    seed: int,
) -> None:
    if model.settings.source is None:
        raise ValueError("Model settings do not contain an eigenvalue source")

    settings = copy.deepcopy(model.settings)
    settings.run_mode = "eigenvalue"
    settings.particles = transport.particles
    settings.batches = transport.batches
    settings.inactive = transport.inactive
    settings.seed = seed
    settings.temperature = {"method": "interpolation"}
    settings.output = {"summary": False, "tallies": False}
    model.settings = settings


def case_seed(base_seed: int, case_index: int, replicate: int) -> int:
    return int(base_seed + case_index * 10_003 + replicate * 1_009)


def run_case(
    source_model: openmc.Model,
    case: CaseDefinition,
    *,
    case_index: int,
    replicate: int,
    transport: TransportSettings,
    run_root: Path,
) -> CaseResult:
    model = copy.deepcopy(source_model)
    apply_case_to_model(model, case)
    if not transport.keep_tallies:
        remove_mesh_tallies(model)

    seed = case_seed(transport.base_seed, case_index, replicate)
    configure_run_settings(model, transport, seed=seed)

    run_dir = run_root / "runs" / case.case_id / f"replicate_{replicate:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    statepoint_path = Path(
        model.run(
            cwd=run_dir,
            threads=transport.threads,
            openmc_exec=transport.openmc_exec,
            export_model_xml=True,
        )
    ).resolve()

    with openmc.StatePoint(statepoint_path) as statepoint:
        k_eff = float(statepoint.keff.n)
        k_eff_std = float(statepoint.keff.s)

    rho = rho_pcm(k_eff)
    return CaseResult(
        case_id=case.case_id,
        coefficient_name=case.coefficient_name,
        case_name=case.case_name,
        replicate=replicate,
        seed=seed,
        material_name=case.material_name,
        kind=case.kind,
        baseline_value=case.baseline_value,
        perturbed_value=case.perturbed_value,
        variable_unit=case.variable_unit,
        k_eff=k_eff,
        k_eff_std=k_eff_std,
        rho_pcm=rho,
        rho_std_pcm=rho_std_pcm(k_eff, k_eff_std),
        run_dir=str(run_dir.resolve()),
        statepoint_path=str(statepoint_path),
    )


def _combined_mean_std(values: list[float], std_devs: list[float]) -> tuple[float, float]:
    if len(values) != len(std_devs):
        raise ValueError("values and std_devs must have the same length")
    if not values:
        raise ValueError("Cannot average an empty result set")

    count = len(values)
    mean = sum(values) / count
    measurement_variance = sum(std_dev * std_dev for std_dev in std_devs) / (
        count * count
    )
    sample_variance = 0.0
    if count > 1:
        sample_variance = (
            sum((value - mean) ** 2 for value in values) / (count - 1) / count
        )
    return mean, math.sqrt(measurement_variance + sample_variance)


def average_case_results(results: list[CaseResult]) -> dict[str, AveragedCaseResult]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case_id, []).append(result)

    averaged: dict[str, AveragedCaseResult] = {}
    for case_id, case_results in grouped.items():
        first = case_results[0]
        k_eff, k_eff_std = _combined_mean_std(
            [result.k_eff for result in case_results],
            [result.k_eff_std for result in case_results],
        )
        rho, rho_std = _combined_mean_std(
            [result.rho_pcm for result in case_results],
            [result.rho_std_pcm for result in case_results],
        )
        averaged[case_id] = AveragedCaseResult(
            case_id=case_id,
            coefficient_name=first.coefficient_name,
            case_name=first.case_name,
            material_name=first.material_name,
            kind=first.kind,
            baseline_value=first.baseline_value,
            perturbed_value=first.perturbed_value,
            variable_unit=first.variable_unit,
            replicate_count=len(case_results),
            k_eff=k_eff,
            k_eff_std=k_eff_std,
            rho_pcm=rho,
            rho_std_pcm=rho_std,
        )
    return averaged


def compute_coefficient_results(
    definitions: list[CoefficientDefinition],
    averaged_cases: dict[str, AveragedCaseResult],
) -> list[CoefficientResult]:
    base = averaged_cases.get("base")
    coefficient_results: list[CoefficientResult] = []
    for definition in definitions:
        minus = averaged_cases[f"{definition.name}_minus"]
        plus = averaged_cases[f"{definition.name}_plus"]
        denominator = 2.0 * definition.delta_value
        coefficient = (plus.rho_pcm - minus.rho_pcm) / denominator
        coefficient_std = (
            math.sqrt(plus.rho_std_pcm**2 + minus.rho_std_pcm**2) / denominator
        )

        per_percent = None
        per_percent_std = None
        if definition.kind == "density":
            delta_percent = definition.delta_value / definition.baseline_value * 100.0
            per_percent = (plus.rho_pcm - minus.rho_pcm) / (2.0 * delta_percent)
            per_percent_std = (
                math.sqrt(plus.rho_std_pcm**2 + minus.rho_std_pcm**2)
                / (2.0 * delta_percent)
            )

        coefficient_results.append(
            CoefficientResult(
                name=definition.name,
                label=definition.label,
                material_name=definition.material_name,
                kind=definition.kind,
                baseline_value=definition.baseline_value,
                minus_value=definition.baseline_value - definition.delta_value,
                plus_value=definition.baseline_value + definition.delta_value,
                delta_value=definition.delta_value,
                variable_unit=definition.variable_unit,
                coefficient_value=coefficient,
                coefficient_std=coefficient_std,
                coefficient_unit=definition.coefficient_unit,
                base_rho_pcm=None if base is None else base.rho_pcm,
                base_rho_std_pcm=None if base is None else base.rho_std_pcm,
                minus_rho_pcm=minus.rho_pcm,
                minus_rho_std_pcm=minus.rho_std_pcm,
                plus_rho_pcm=plus.rho_pcm,
                plus_rho_std_pcm=plus.rho_std_pcm,
                coefficient_per_percent_density=per_percent,
                coefficient_per_percent_density_std=per_percent_std,
            )
        )
    return coefficient_results


def _format_float(value: float | None, digits: int = 10) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def write_case_results_csv(results: list[CaseResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "case_results.csv"
    fieldnames = [
        "case_id",
        "coefficient_name",
        "case_name",
        "replicate",
        "seed",
        "material_name",
        "kind",
        "baseline_value",
        "perturbed_value",
        "variable_unit",
        "k_eff",
        "k_eff_std",
        "rho_pcm",
        "rho_std_pcm",
        "run_dir",
        "statepoint_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for key in (
                "baseline_value",
                "perturbed_value",
                "k_eff",
                "k_eff_std",
                "rho_pcm",
                "rho_std_pcm",
            ):
                row[key] = _format_float(row[key])
            writer.writerow(row)
    return csv_path


def write_coefficient_results_csv(
    results: list[CoefficientResult], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reactivity_coefficients.csv"
    fieldnames = [
        "name",
        "label",
        "material_name",
        "kind",
        "baseline_value",
        "minus_value",
        "plus_value",
        "delta_value",
        "variable_unit",
        "coefficient_value",
        "coefficient_std",
        "coefficient_unit",
        "coefficient_per_percent_density",
        "coefficient_per_percent_density_std",
        "base_rho_pcm",
        "base_rho_std_pcm",
        "minus_rho_pcm",
        "minus_rho_std_pcm",
        "plus_rho_pcm",
        "plus_rho_std_pcm",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for key, value in row.items():
                if isinstance(value, float) or value is None:
                    row[key] = _format_float(value)
            writer.writerow(row)
    return csv_path


def write_reactivity_plot(
    definitions: list[CoefficientDefinition],
    averaged_cases: dict[str, AveragedCaseResult],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "reactivity_coefficients.png"
    base = averaged_cases.get("base")
    fig, axes = plt.subplots(1, len(definitions), figsize=(14, 4.2))
    if len(definitions) == 1:
        axes = [axes]

    for axis, definition in zip(axes, definitions):
        minus = averaged_cases[f"{definition.name}_minus"]
        plus = averaged_cases[f"{definition.name}_plus"]

        if definition.kind == "density":
            delta_percent = definition.delta_value / definition.baseline_value * 100.0
            x_values = [-delta_percent, delta_percent]
            x_label = "D2O density change [%]"
        else:
            x_values = [minus.perturbed_value, plus.perturbed_value]
            x_label = f"{definition.label} [{definition.variable_unit}]"

        y_values = [minus.rho_pcm, plus.rho_pcm]
        y_errors = [minus.rho_std_pcm, plus.rho_std_pcm]
        axis.errorbar(x_values, y_values, yerr=y_errors, marker="o", capsize=4)

        if base is not None:
            base_x = 0.0 if definition.kind == "density" else definition.baseline_value
            axis.errorbar(
                [base_x],
                [base.rho_pcm],
                yerr=[base.rho_std_pcm],
                marker="s",
                capsize=4,
                color="tab:orange",
            )

        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(definition.label)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Reactivity [pcm]")
        axis.grid(True, alpha=0.25)

    fig.suptitle("Concentric reactor reactivity perturbations", fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def write_run_summary_json(
    *,
    output_dir: Path,
    model_xml_path: Path,
    transport: TransportSettings,
    baselines: dict[str, MaterialBaseline],
    definitions: list[CoefficientDefinition],
    cases: list[CaseDefinition],
    case_results: list[CaseResult],
    averaged_cases: dict[str, AveragedCaseResult],
    coefficient_results: list[CoefficientResult],
    files: dict[str, Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    listed_files = {**files, "run_summary_json": summary_path}
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "openmc": {
            "version": openmc.__version__,
            "cross_sections": openmc.config.get("cross_sections"),
            "executable": transport.openmc_exec,
        },
        "source_model": {
            "model_xml_path": str(model_xml_path),
            "sha256": file_sha256(model_xml_path),
        },
        "transport": asdict(transport),
        "material_baselines": baselines,
        "coefficient_definitions": definitions,
        "cases": cases,
        "case_results": case_results,
        "averaged_cases": averaged_cases,
        "coefficients": coefficient_results,
        "files": {name: str(path) for name, path in listed_files.items()},
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def publish_results(files: dict[str, Path], reference_results_dir: Path) -> dict[str, Path]:
    reference_results_dir.mkdir(parents=True, exist_ok=True)
    published: dict[str, Path] = {}
    for name, source in files.items():
        destination = reference_results_dir / source.name
        shutil.copy2(source, destination)
        published[name] = destination
    return published


def inspect_run_plan(
    *,
    model_xml_path: Path,
    baselines: dict[str, MaterialBaseline],
    definitions: list[CoefficientDefinition],
    cases: list[CaseDefinition],
    transport: TransportSettings,
) -> dict[str, Any]:
    return {
        "model_xml_path": str(model_xml_path),
        "model_sha256": file_sha256(model_xml_path),
        "transport": asdict(transport),
        "material_baselines": baselines,
        "coefficient_definitions": definitions,
        "cases": cases,
        "case_count": len(cases),
    }


def run_reactivity_coefficients(
    *,
    model_xml_path: Path,
    output_dir: Path,
    transport: TransportSettings,
    fuel_temperature_delta_k: float,
    moderator_temperature_delta_k: float,
    moderator_density_delta_fraction: float,
    replicates: int,
    publish_reference_data: bool,
    reference_results_dir: Path,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not openmc.config.get("cross_sections"):
        raise RuntimeError(
            "Set openmc.config['cross_sections'] or OPENMC_CROSS_SECTIONS before "
            "running reactivity coefficient calculations."
        )

    source_model = load_source_model(model_xml_path)
    baselines = collect_material_baselines(source_model)
    definitions = build_coefficient_definitions(
        baselines,
        fuel_temperature_delta_k=fuel_temperature_delta_k,
        moderator_temperature_delta_k=moderator_temperature_delta_k,
        moderator_density_delta_fraction=moderator_density_delta_fraction,
    )
    cases = build_case_definitions(definitions)

    output_dir.mkdir(parents=True, exist_ok=True)
    case_results: list[CaseResult] = []
    for case_index, case in enumerate(cases):
        for replicate in range(1, replicates + 1):
            print(
                f"Running {case.case_id} replicate {replicate}/{replicates} "
                f"({transport.particles} particles x {transport.batches} batches)"
            )
            result = run_case(
                source_model,
                case,
                case_index=case_index,
                replicate=replicate,
                transport=transport,
                run_root=output_dir,
            )
            case_results.append(result)
            print(
                f"  k_eff = {result.k_eff:.6f} +/- {result.k_eff_std:.6f}; "
                f"rho = {result.rho_pcm:+.1f} +/- {result.rho_std_pcm:.1f} pcm"
            )

    averaged_cases = average_case_results(case_results)
    coefficient_results = compute_coefficient_results(definitions, averaged_cases)

    files = {
        "case_results_csv": write_case_results_csv(case_results, output_dir),
        "reactivity_coefficients_csv": write_coefficient_results_csv(
            coefficient_results, output_dir
        ),
        "reactivity_coefficients_png": write_reactivity_plot(
            definitions, averaged_cases, output_dir
        ),
    }
    files["run_summary_json"] = write_run_summary_json(
        output_dir=output_dir,
        model_xml_path=model_xml_path,
        transport=transport,
        baselines=baselines,
        definitions=definitions,
        cases=cases,
        case_results=case_results,
        averaged_cases=averaged_cases,
        coefficient_results=coefficient_results,
        files=files,
    )

    published_files: dict[str, Path] = {}
    if publish_reference_data:
        published_files = publish_results(files, reference_results_dir)

    return {
        "output_dir": output_dir,
        "files": files,
        "published_files": published_files,
        "coefficients": coefficient_results,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run finite-difference reactivity coefficient scans for the "
            "unrodded concentric OpenMC model."
        )
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_XML_PATH),
        help="Path to model.xml or a directory containing model.xml.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for run files and result artifacts.",
    )
    parser.add_argument(
        "--reference-results-dir",
        default=str(DEFAULT_REFERENCE_RESULTS_DIR),
        help="Reference-data destination used only with --publish-reference-data.",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use the production particle/batch preset unless overridden.",
    )
    parser.add_argument("--particles", type=positive_int)
    parser.add_argument("--batches", type=positive_int)
    parser.add_argument("--inactive", type=positive_int)
    parser.add_argument("--replicates", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int)
    parser.add_argument("--openmc-exec")
    parser.add_argument("--base-seed", type=positive_int, default=91_231)
    parser.add_argument(
        "--fuel-temperature-delta-k",
        type=positive_float,
        default=50.0,
    )
    parser.add_argument(
        "--moderator-temperature-delta-k",
        type=positive_float,
        default=10.0,
    )
    parser.add_argument(
        "--moderator-density-delta-fraction",
        type=positive_float,
        default=0.01,
        help="Fractional D2O density perturbation; 0.01 means +/-1 percent.",
    )
    parser.add_argument(
        "--keep-tallies",
        action="store_true",
        help="Keep tallies from the source model instead of removing mesh tallies.",
    )
    parser.add_argument(
        "--publish-reference-data",
        action="store_true",
        help="Copy result CSV/JSON/PNG artifacts into reference_data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the model and print the planned cases without running OpenMC.",
    )
    return parser


def transport_from_args(args: argparse.Namespace) -> TransportSettings:
    if args.production:
        particles_default = PRODUCTION_PARTICLES
        batches_default = PRODUCTION_BATCHES
        inactive_default = PRODUCTION_INACTIVE
    else:
        particles_default = FAST_PARTICLES
        batches_default = FAST_BATCHES
        inactive_default = FAST_INACTIVE

    return TransportSettings(
        particles=args.particles or particles_default,
        batches=args.batches or batches_default,
        inactive=args.inactive or inactive_default,
        threads=args.threads or (os.cpu_count() or 1),
        openmc_exec=resolve_openmc_exec(args.openmc_exec),
        base_seed=args.base_seed,
        keep_tallies=bool(args.keep_tallies),
    )


def validate_transport_settings(transport: TransportSettings) -> None:
    if transport.inactive >= transport.batches:
        raise ValueError("inactive batches must be smaller than total batches")


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    model_xml_path = resolve_model_xml_path(args.model)
    output_dir = Path(args.output_dir).expanduser().resolve()
    reference_results_dir = Path(args.reference_results_dir).expanduser().resolve()
    transport = transport_from_args(args)
    try:
        validate_transport_settings(transport)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        source_model = load_source_model(model_xml_path)
        baselines = collect_material_baselines(source_model)
        definitions = build_coefficient_definitions(
            baselines,
            fuel_temperature_delta_k=args.fuel_temperature_delta_k,
            moderator_temperature_delta_k=args.moderator_temperature_delta_k,
            moderator_density_delta_fraction=args.moderator_density_delta_fraction,
        )
        cases = build_case_definitions(definitions)
        plan = inspect_run_plan(
            model_xml_path=model_xml_path,
            baselines=baselines,
            definitions=definitions,
            cases=cases,
            transport=transport,
        )
        print(json.dumps(_json_safe(plan), indent=2, sort_keys=True))
        return 0

    result = run_reactivity_coefficients(
        model_xml_path=model_xml_path,
        output_dir=output_dir,
        transport=transport,
        fuel_temperature_delta_k=args.fuel_temperature_delta_k,
        moderator_temperature_delta_k=args.moderator_temperature_delta_k,
        moderator_density_delta_fraction=args.moderator_density_delta_fraction,
        replicates=args.replicates,
        publish_reference_data=bool(args.publish_reference_data),
        reference_results_dir=reference_results_dir,
    )

    print(f"Results written to {result['output_dir']}")
    for coefficient in result["coefficients"]:
        print(
            f"{coefficient.label}: "
            f"{coefficient.coefficient_value:+.3f} +/- "
            f"{coefficient.coefficient_std:.3f} {coefficient.coefficient_unit}"
        )
        if coefficient.coefficient_per_percent_density is not None:
            print(
                "  "
                f"{coefficient.coefficient_per_percent_density:+.3f} +/- "
                f"{coefficient.coefficient_per_percent_density_std:.3f} "
                "pcm/%density"
            )
    if result["published_files"]:
        print(f"Published reference artifacts to {reference_results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
