from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
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
from openmc_to_genfoam_xs import (
    DEFAULT_OUTPUT_SUBDIR as DEFAULT_GENFOAM_XS_OUTPUT_SUBDIR,
    build_nuclear_data_text_from_export,
    generate_genfoam_xs,
    select_scatter_matrix_payload,
)


CASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CASE_DIR.parent
DEFAULT_MGXS_EXPORT_DIR = PROJECT_DIR / "openmc" / "build" / "concentric" / "mgxs_export"
DEFAULT_OUTPUT_DIR = CASE_DIR / "constant" / "generated"
CM_TO_M = 1.0e-2
EV_TO_J = 1.602176487e-19


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _progress(message: str) -> None:
    print(f"[prepare_concentric_case] {message}", file=sys.stderr, flush=True)


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
    chi_payload = group_constants.get("chi-prompt", group_constants["chi"])
    chi = [float(value) for value in chi_payload["mean"]]
    _scatter_name, scatter_payload = select_scatter_matrix_payload(group_constants)
    scatter_matrix = _collapse_legendre_moment(scatter_payload["mean"])

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
        if xs_type in {"scatter matrix", "nu-scatter matrix", "consistent nu-scatter matrix"}:
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
        if xs_type == "kappa-fission":
            converted[xs_type] = {
                "mean": _deep_scale(mean_value, EV_TO_J / CM_TO_M),
                "std_dev": _deep_scale(std_dev_value, EV_TO_J / CM_TO_M),
                "units": "J/m",
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


def _vol_scalar_header(object_name: str) -> str:
    return _foam_header("volScalarField", object_name, version="2")

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
    blocks = case_spec["mesh"].get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("Expected case_spec['mesh']['blocks'] in the generated Gmsh mesh manifest")

    counts: dict[str, int] = {}
    for block in blocks:
        region = block["region"]
        counts[region] = counts.get(region, 0) + 1
    return counts


def build_nuclear_data_text(case_spec: dict[str, Any]) -> str:
    zone_counts = _zone_block_counts(case_spec)
    return build_nuclear_data_text_from_export(
        xs_payload=case_spec["genfoam_xs"],
        domain_order=list(case_spec["config"]["domain_order"]),
        zone_counts=zone_counts,
    )


def write_generated_case_files(case_spec: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    group_count = int(case_spec["config"]["group_count"])
    output_paths = {
        "nuclearData": case_dir / "constant" / "neutroRegion" / "nuclearData",
    }

    write_text(output_paths["nuclearData"], build_nuclear_data_text(case_spec))

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


def build_case_spec(
    mgxs_export_dir: Path,
    output_dir: Path,
    rerun_mgxs: bool = False,
    openmc_particles: int | None = None,
    openmc_batches: int | None = None,
    openmc_inactive: int | None = None,
    openmc_threads: int | None = None,
    legendre_order: int | None = None,
) -> dict[str, Any]:
    stage_start = time.perf_counter()
    _progress(f"Loading MGXS export from {mgxs_export_dir}")
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
    _progress(
        "Loaded MGXS metadata: "
        f"{len(export['config']['energy_group_edges_ev']) - 1} groups, "
        f"P{export['config']['legendre_order']}, "
        f"{len(export['config']['domain_definitions'])} regions, "
        f"{len(mesh_info['blocks'])} mesh blocks "
        f"({len(mesh_info['radial_edges_m']) - 1} radial x {len(mesh_info['axial_edges_m']) - 1} axial intervals) "
        f"in {_format_elapsed(time.perf_counter() - stage_start)}"
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
    stage_start = time.perf_counter()
    xs_mode = "rerun" if rerun_mgxs else "existing export"
    _progress(f"Building GeN-Foam XS payload from {xs_mode}")
    genfoam_xs_output_dir = output_dir / DEFAULT_GENFOAM_XS_OUTPUT_SUBDIR
    genfoam_xs = generate_genfoam_xs(
        mgxs_export_dir=mgxs_export_dir,
        output_dir=genfoam_xs_output_dir,
        rerun_mgxs=rerun_mgxs,
        particles=openmc_particles,
        batches=openmc_batches,
        inactive=openmc_inactive,
        threads=openmc_threads,
        legendre_order=legendre_order,
    )
    reference_run = genfoam_xs["reference_run"]
    _progress(
        "Built XS payload in "
        f"{_format_elapsed(time.perf_counter() - stage_start)} "
        f"(source keff={reference_run['keff']}, "
        f"promptGenerationTime={reference_run['prompt_generation_time_s']:.12g}s, "
        f"mode={genfoam_xs['source']['mode']})"
    )

    return {
        "source": {
            "mgxs_export_dir": str(mgxs_export_dir),
            "model_xml_path": str(model_xml_path),
            "mgxs_constants_json": str(mgxs_json_path),
            "genfoam_xs_summary": genfoam_xs["files"]["summary"],
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
        "genfoam_xs": genfoam_xs,
    }


def write_case_outputs(case_spec: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": output_dir / "concentric_case_manifest.json",
        "mesh_manifest": output_dir / "concentric_reactor_mesh_manifest.json",
        "mesh_file": output_dir / "concentric_reactor_wedge.msh",
        "zones_csv": output_dir / "concentric_mesh_regions.csv",
        "materials": output_dir / "concentric_materials.json",
        "genfoam_xs_summary": Path(case_spec["genfoam_xs"]["files"]["summary"]),
        "genfoam_xs_adapted_nuclear_data": Path(case_spec["genfoam_xs"]["files"]["adapted_nuclear_data"]),
    }
    stage_start = time.perf_counter()
    _progress(f"Generating wedge mesh at {files['mesh_file']}")
    write_mesh_assets(
        mesh_file=files["mesh_file"],
        manifest_file=files["mesh_manifest"],
        mesh_kind=WEDGE_MESH_KIND,
    )
    _progress(f"Generated mesh in {_format_elapsed(time.perf_counter() - stage_start)}")
    stage_start = time.perf_counter()
    _progress(f"Writing case metadata under {output_dir}")
    write_json(files["manifest"], case_spec)
    write_json(files["materials"], {"materials": case_spec["materials"], "config": case_spec["config"]})
    write_csv(
        files["zones_csv"],
        rows=case_spec["mesh_regions"],
        fieldnames=["region", "cell_zone"],
    )
    _progress(f"Wrote case metadata in {_format_elapsed(time.perf_counter() - stage_start)}")
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
    parser.add_argument(
        "--rerun-mgxs",
        action="store_true",
        help="Re-run the MGXS export through openmc/mgxs_export.py instead of using the existing mgxs_constants.json",
    )
    parser.add_argument(
        "--openmc-particles",
        type=int,
        default=None,
        help="Optional override for the MGXS rerun particle count",
    )
    parser.add_argument(
        "--openmc-batches",
        type=int,
        default=None,
        help="Optional override for the MGXS rerun batch count",
    )
    parser.add_argument(
        "--openmc-inactive",
        type=int,
        default=None,
        help="Optional override for the MGXS rerun inactive batch count",
    )
    parser.add_argument(
        "--openmc-threads",
        type=int,
        default=None,
        help="Optional override for the MGXS rerun thread count",
    )
    parser.add_argument(
        "--legendre-order",
        type=int,
        default=None,
        help="Optional Legendre scattering order for the MGXS rerun path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    rerun_mgxs = args.rerun_mgxs or any(
        value is not None
        for value in (
            args.openmc_particles,
            args.openmc_batches,
            args.openmc_inactive,
            args.openmc_threads,
            args.legendre_order,
        )
    )
    _progress(
        "Starting case preparation "
        f"(mgxs_export_dir={args.mgxs_export_dir}, output_dir={args.output_dir}, rerun_mgxs={rerun_mgxs})"
    )
    case_spec = build_case_spec(
        args.mgxs_export_dir,
        args.output_dir,
        rerun_mgxs=rerun_mgxs,
        openmc_particles=args.openmc_particles,
        openmc_batches=args.openmc_batches,
        openmc_inactive=args.openmc_inactive,
        openmc_threads=args.openmc_threads,
        legendre_order=args.legendre_order,
    )
    files = write_case_outputs(case_spec, args.output_dir)
    stage_start = time.perf_counter()
    _progress(f"Writing OpenFOAM case files into {CASE_DIR}")
    generated_case_files = write_generated_case_files(case_spec, CASE_DIR)
    _progress(f"Wrote OpenFOAM case files in {_format_elapsed(time.perf_counter() - stage_start)}")

    rendered_generated_case_files: dict[str, Any] = {}
    for name, value in generated_case_files.items():
        if isinstance(value, list):
            rendered_generated_case_files[name] = value
        else:
            rendered_generated_case_files[name] = str(value)

    print(
        json.dumps(
            {
                "group_count": case_spec["config"]["group_count"],
                "mesh_region_count": len(case_spec["mesh_regions"]),
                "xs_source_keff": case_spec["genfoam_xs"]["reference_run"]["keff"],
                "output_files": {name: str(path) for name, path in files.items()},
                "generated_case_files": rendered_generated_case_files,
            },
            indent=2,
        )
    )
    _progress(f"Completed case preparation in {_format_elapsed(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
