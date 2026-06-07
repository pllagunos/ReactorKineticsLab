from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CASE_DIR = Path(__file__).resolve().parent
FIELD_TOKEN_PATTERN = re.compile(r"internalField\s+(uniform|nonuniform)\s+([^;]*);", re.DOTALL)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_time_directories(case_dir: Path) -> list[Path]:
    time_dirs: list[tuple[float, Path]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            time_value = float(child.name)
        except ValueError:
            continue
        time_dirs.append((time_value, child))
    return [path for _, path in sorted(time_dirs)]


def resolve_time_directory(case_dir: Path, requested_time: str | None) -> Path:
    if requested_time is not None:
        candidate = case_dir / requested_time
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find time directory: {candidate}")
        return candidate

    time_dirs = numeric_time_directories(case_dir)
    if not time_dirs:
        raise FileNotFoundError(f"No numeric time directories found in {case_dir}")
    return time_dirs[-1]


def read_scalar_field(field_path: Path) -> np.ndarray:
    text = field_path.read_text(encoding="utf-8")
    match = FIELD_TOKEN_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not parse internalField from {field_path}")

    kind = match.group(1)
    payload = match.group(2).strip()
    if kind == "uniform":
        value = float(payload.split()[0])
        return np.asarray([value], dtype=float)

    list_match = re.search(r"List<scalar>\s+(\d+)\s*\((.*?)\)\s*$", payload, re.DOTALL)
    if list_match is None:
        raise ValueError(f"Could not parse nonuniform scalar list from {field_path}")
    expected_count = int(list_match.group(1))
    values = np.fromstring(list_match.group(2), sep=" ", dtype=float)
    if values.size != expected_count:
        raise ValueError(
            f"Field {field_path} declared {expected_count} values but parsed {values.size}"
        )
    return values


def structured_mesh(manifest: dict[str, Any]) -> dict[str, Any]:
    mesh = manifest["mesh"]
    mesh_kind = mesh.get("kind")
    if mesh_kind not in {"axisymmetric-wedge", "full-3d-experimental"}:
        raise NotImplementedError(
            "plot_results.py expects a structured Gmsh mesh manifest from the current genfoam workflow. "
            f"Found mesh.kind={mesh_kind!r}."
        )
    if "blocks" not in mesh or "radial_edges_m" not in mesh or "axial_edges_m" not in mesh:
        raise ValueError("Structured Gmsh mesh metadata is missing blocks or grid edges")
    return mesh


def build_grid(manifest: dict[str, Any], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = structured_mesh(manifest)
    blocks = mesh["blocks"]
    if values.size != len(blocks):
        raise ValueError(
            f"Field value count {values.size} does not match manifest block count {len(blocks)}"
        )

    radial_edges = np.asarray(mesh["radial_edges_m"], dtype=float)
    axial_edges = np.asarray(mesh["axial_edges_m"], dtype=float)
    grid = np.full((axial_edges.size - 1, radial_edges.size - 1), np.nan, dtype=float)

    radial_lookup = {round(float(edge), 12): index for index, edge in enumerate(radial_edges)}
    axial_lookup = {round(float(edge), 12): index for index, edge in enumerate(axial_edges)}
    for value, block in zip(values, blocks, strict=True):
        r_min = radial_lookup[round(float(block["r_min_m"]), 12)]
        z_min = axial_lookup[round(float(block["z_min_m"]), 12)]
        grid[z_min, r_min] = float(value)
    return radial_edges, axial_edges, grid


def volume_weighted_profiles(manifest: dict[str, Any], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blocks = structured_mesh(manifest)["blocks"]
    radial_centers: list[float] = []
    axial_centers: list[float] = []
    radial_weights: list[float] = []
    axial_weights: list[float] = []
    radial_values: list[float] = []
    axial_values: list[float] = []

    for value, block in zip(values, blocks, strict=True):
        r_min = float(block["r_min_m"])
        r_max = float(block["r_max_m"])
        z_min = float(block["z_min_m"])
        z_max = float(block["z_max_m"])
        cylindrical_weight = 0.5 * (r_max**2 - r_min**2) * (z_max - z_min)
        radial_centers.append(0.5 * (r_min + r_max))
        axial_centers.append(0.5 * (z_min + z_max))
        radial_weights.append(cylindrical_weight)
        axial_weights.append(cylindrical_weight)
        radial_values.append(float(value))
        axial_values.append(float(value))

    def aggregate(centers: list[float], sample_values: list[float], sample_weights: list[float]) -> tuple[np.ndarray, np.ndarray]:
        accum: dict[float, tuple[float, float]] = {}
        for center, sample_value, sample_weight in zip(centers, sample_values, sample_weights, strict=True):
            current_value, current_weight = accum.get(center, (0.0, 0.0))
            accum[center] = (
                current_value + sample_value * sample_weight,
                current_weight + sample_weight,
            )
        xs = np.asarray(sorted(accum), dtype=float)
        ys = np.asarray([accum[x][0] / accum[x][1] for x in xs], dtype=float)
        return xs, ys

    radial_x, radial_y = aggregate(radial_centers, radial_values, radial_weights)
    axial_x, axial_y = aggregate(axial_centers, axial_values, axial_weights)
    return radial_x, radial_y, axial_x, axial_y


def available_flux_fields(time_dir: Path) -> list[str]:
    region_dir = time_dir / "neutroRegion"
    fields = [child.name for child in region_dir.iterdir() if child.is_file() and re.fullmatch(r"flux\d+", child.name)]
    return sorted(fields, key=lambda name: int(name[4:]))


def default_fields_to_plot(time_dir: Path) -> list[str]:
    candidates = ["oneGroupFlux", "powerDensity"]
    flux_fields = available_flux_fields(time_dir)
    if flux_fields:
        candidates.append(flux_fields[0])
        if flux_fields[-1] != flux_fields[0]:
            candidates.append(flux_fields[-1])
    return [field for field in candidates if (time_dir / "neutroRegion" / field).exists()]


def plot_field(ax: plt.Axes, manifest: dict[str, Any], field_name: str, values: np.ndarray) -> None:
    radial_edges, axial_edges, grid = build_grid(manifest, values)
    mesh = ax.pcolormesh(radial_edges, axial_edges, grid, shading="flat", cmap="viridis")
    ax.set_title(field_name)
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    plt.colorbar(mesh, ax=ax, shrink=0.85)


def plot_one_group_profiles(ax_radial: plt.Axes, ax_axial: plt.Axes, manifest: dict[str, Any], values: np.ndarray) -> None:
    radial_x, radial_y, axial_x, axial_y = volume_weighted_profiles(manifest, values)
    ax_radial.plot(radial_x, radial_y, marker="o", linewidth=1.5)
    ax_radial.set_title("One-Group Flux Radial Profile")
    ax_radial.set_xlabel("r [m]")
    ax_radial.set_ylabel("Flux")
    ax_radial.grid(True, alpha=0.3)

    ax_axial.plot(axial_x, axial_y, marker="o", linewidth=1.5)
    ax_axial.set_title("One-Group Flux Axial Profile")
    ax_axial.set_xlabel("z [m]")
    ax_axial.set_ylabel("Flux")
    ax_axial.grid(True, alpha=0.3)


def generate_plot(case_dir: Path, time_dir: Path, fields: list[str], output_path: Path, show: bool) -> None:
    manifest = load_json(case_dir / "constant" / "generated" / "concentric_case_manifest.json")
    one_group_path = time_dir / "neutroRegion" / "oneGroupFlux"
    one_group_values = read_scalar_field(one_group_path)

    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid_spec = figure.add_gridspec(2, 3)

    heatmap_axes = [
        figure.add_subplot(grid_spec[0, 0]),
        figure.add_subplot(grid_spec[0, 1]),
        figure.add_subplot(grid_spec[0, 2]),
        figure.add_subplot(grid_spec[1, 0]),
    ]
    profile_ax_radial = figure.add_subplot(grid_spec[1, 1])
    profile_ax_axial = figure.add_subplot(grid_spec[1, 2])

    for ax, field_name in zip(heatmap_axes, fields, strict=False):
        field_values = read_scalar_field(time_dir / "neutroRegion" / field_name)
        plot_field(ax, manifest, field_name, field_values)

    for unused_ax in heatmap_axes[len(fields):]:
        unused_ax.axis("off")

    plot_one_group_profiles(profile_ax_radial, profile_ax_axial, manifest, one_group_values)
    figure.suptitle(f"GeN-Foam Results at t = {time_dir.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot flux-relevant GeN-Foam outputs for the concentric reactor case.")
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR, help="Path to the GeN-Foam case directory")
    parser.add_argument("--time", dest="time_name", default=None, help="Numeric time directory to plot. Defaults to the latest time.")
    parser.add_argument(
        "--fields",
        default=None,
        help="Comma-separated list of scalar fields under <time>/neutroRegion to plot. Defaults to oneGroupFlux,powerDensity,flux0,last flux if present.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to the output PNG. Defaults to plots/results_<time>.png")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    time_dir = resolve_time_directory(case_dir, args.time_name)

    if args.fields is None:
        fields = default_fields_to_plot(time_dir)
    else:
        fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    if not fields:
        raise ValueError("No fields selected for plotting")
    if len(fields) > 4:
        raise ValueError("Plot at most 4 heatmap fields at once")

    for field_name in fields:
        field_path = time_dir / "neutroRegion" / field_name
        if not field_path.exists():
            raise FileNotFoundError(f"Could not find field {field_name!r} at {field_path}")

    output_path = args.output.resolve() if args.output is not None else case_dir / "plots" / f"results_{time_dir.name}.png"
    generate_plot(case_dir, time_dir, fields, output_path, show=args.show)
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
