import csv
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc

import fuel_element
import concentric_fuel
import reactor_geometry
import build_simulation
from ploting import resolve_openmc_exec

importlib.reload(fuel_element)
importlib.reload(concentric_fuel)
importlib.reload(reactor_geometry)
importlib.reload(build_simulation)

from concentric_fuel import (
    ConcentricElementParameters,
    build_parameter_report,
    validate_parameters,
 )
from reactor_geometry import ReactorTankParameters, validate_reactor_tanks

CONCENTRIC_FUEL_PARAMETERS = ConcentricElementParameters(
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

REACTOR_TANK_PARAMETERS = ReactorTankParameters(
    d2o_tank_radius_cm=250.0,
    h2o_tank_radius_cm=500.0, 
    h_d2o_tank_cm=600.0,
    h_h2o_tank_cm=1000.0, #700
 )

validate_parameters(CONCENTRIC_FUEL_PARAMETERS)
validate_reactor_tanks(CONCENTRIC_FUEL_PARAMETERS, REACTOR_TANK_PARAMETERS)

GEOMETRY = build_parameter_report(CONCENTRIC_FUEL_PARAMETERS)
GEOMETRY.update(REACTOR_TANK_PARAMETERS.to_geometry_dict())

GLOBAL_MESH_SHAPE = (160, 160)
FUEL_MESH_SHAPE = (220, 220)
DEFAULT_OPENMC_THREADS = os.cpu_count() or 1
OPENMC_THREADS = int(os.environ.get("OPENMC_THREADS", DEFAULT_OPENMC_THREADS))
OPENMC_EXEC = resolve_openmc_exec()

RUN_DIR = Path("build") / "concentric"
RUN_DIR.mkdir(parents=True, exist_ok=True)

GEOMETRY_REPORT = {
    "fuel_element": build_parameter_report(CONCENTRIC_FUEL_PARAMETERS),
    "reactor_tanks": REACTOR_TANK_PARAMETERS.to_geometry_dict(),
    "openmc_threads": OPENMC_THREADS,
    "openmc_exec": OPENMC_EXEC,
}
GEOMETRY_REPORT

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def rho_pcm(k: float) -> float:
    """Reactivity in pcm from k_eff."""
    return (k - 1.0) / k * 1e5


def estimate_critical_insertion(
    x_values: np.ndarray, rho_total_pcm: np.ndarray,
) -> float | None:
    """Linear interpolation of the insertion fraction where ρ crosses zero."""
    for j in range(len(x_values) - 1):
        y0, y1 = rho_total_pcm[j], rho_total_pcm[j + 1]
        if y0 == 0.0:
            return float(x_values[j])
        if y0 * y1 <= 0.0 and y1 != y0:
            return float(
                x_values[j] + (0.0 - y0) * (x_values[j + 1] - x_values[j]) / (y1 - y0)
            )
    if rho_total_pcm[-1] == 0.0:
        return float(x_values[-1])
    return None


def export_plots(fig: plt.Figure, export_dir: Path) -> None:
    plot_path = export_dir / "rod_worth.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.close(fig)


def export_rod_worth_csv(
    x_arr: np.ndarray,
    k_values: np.ndarray,
    k_std_values: np.ndarray,
    rho_total: np.ndarray,
    delta_rho: np.ndarray,
    export_dir: Path,
) -> None:
    csv_path = export_dir / "rod_worth.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "insertion_fraction", "insertion_pct", "k_eff", "k_eff_std",
            "rho_total_pcm", "rho_rod_worth_pcm",
        ])
        for i in range(len(x_arr)):
            writer.writerow([
                f"{x_arr[i]:.4f}",
                f"{x_arr[i] * 100:.1f}",
                f"{k_values[i]:.8f}",
                f"{k_std_values[i]:.8f}",
                f"{rho_total[i]:.3f}",
                f"{delta_rho[i]:.3f}",
            ])
    print(f"Rod worth table saved to {csv_path}")


def export_run_summary_json(
    export_dir: Path,
    rod_b10_fraction: float,
    rod_density: float,
    particles: int,
    batches: int,
    inactive: int,
    x_arr: np.ndarray,
    k_values: np.ndarray,
    k_std_values: np.ndarray,
    rho_total: np.ndarray,
    delta_rho: np.ndarray,
    x_crit: float | None,
) -> None:
    json_path = export_dir / "run_summary.json"
    scan_points = []
    for i in range(len(x_arr)):
        scan_points.append({
            "insertion_fraction": round(float(x_arr[i]), 4),
            "insertion_pct": round(float(x_arr[i]) * 100.0, 1),
            "k_eff": float(k_values[i]),
            "k_eff_std": float(k_std_values[i]),
            "rho_total_pcm": round(float(rho_total[i]), 3),
            "rho_rod_worth_pcm": round(float(delta_rho[i]), 3),
        })

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rod_parameters": {
            "b10_fraction": rod_b10_fraction,
            "density_g_per_cm3": rod_density,
        },
        "transport_parameters": {
            "particles": particles,
            "batches": batches,
            "inactive": inactive,
        },
        "results": {
            "rho_clean_pcm": round(float(rho_total[0]), 3),
            "full_insertion_rod_worth_pcm": round(float(delta_rho[-1]), 3),
            "full_insertion_rho_total_pcm": round(float(rho_total[-1]), 3),
            "critical_insertion_pct": (
                round(x_crit * 100.0, 1) if x_crit is not None else None
            ),
        },
        "scan_points": scan_points,
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Run summary saved to {json_path}")


# ── Rod absorber customization ─────────────────────────────────────────────
# Tune these to change the control rod absorption strength.
# Default B4C: B10=3.8, B11=0.2, C=1.0, density=2.52 g/cm³
ROD_B10_FRACTION = 0.001  # wt fraction of B¹⁰ (natural=3.8, enriched up to ~19.9)
ROD_DENSITY = 1.0       # g/cm³

_original_make_materials = fuel_element.make_default_materials

def _rod_scan_make_materials(parameters):
    """Material factory with tunable control-rod absorber strength."""
    materials, material_map = _original_make_materials(parameters)
    control_rod = openmc.Material(name="B4C control rod (custom)")
    control_rod.set_density("g/cm3", ROD_DENSITY)
    control_rod.add_nuclide("B10", ROD_B10_FRACTION)
    control_rod.add_element("C", 4.0)
    control_rod.temperature = 300.0
    material_map["control_rod"] = control_rod
    return openmc.Materials(
        [material_map["fuel"], material_map["moderator"], control_rod]
    ), material_map

# Patch both references so concentric_fuel also picks up the custom material
fuel_element.make_default_materials = _rod_scan_make_materials
concentric_fuel.make_shared_materials = _rod_scan_make_materials

# ── Rod worth scan ──────────────────────────────────────────────────────────
SCAN_PARTICLES = 200000
SCAN_BATCHES = 100
SCAN_INACTIVE = 10

cross_sections = openmc.config.get("cross_sections")
if cross_sections:
    ROD_SCAN_DIR = Path("reference_data") / "concentric" / "rod_scan"
    ROD_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    # Keep build dir for OpenMC run files (statepoints, model.xml) — git-ignored
    OPENMC_RUN_DIR = RUN_DIR / "rod_scan"
    OPENMC_RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Scan 0 → 100 % insertion
    x_arr = np.linspace(0.0, 1.0, 20)
    k_values = np.zeros_like(x_arr)
    k_std_values = np.zeros_like(x_arr)
    rho_total = np.zeros_like(x_arr)

    for i, x_insert in enumerate(x_arr):
        print(f"Rod insertion x = {x_insert:.1f} …")
        model, _ = build_simulation.build_eigenvalue_model(
            CONCENTRIC_FUEL_PARAMETERS,
            REACTOR_TANK_PARAMETERS,
            rod_insertion=x_insert,
            particles=SCAN_PARTICLES,
            batches=SCAN_BATCHES,
            inactive=SCAN_INACTIVE,
            global_mesh_shape=None,
            fuel_mesh_shape=None,
        )
        sp_path = model.run(
            cwd=OPENMC_RUN_DIR, threads=OPENMC_THREADS, openmc_exec=OPENMC_EXEC
        )

        with openmc.StatePoint(sp_path) as sp:
            k_values[i] = sp.keff.n
            k_std_values[i] = sp.keff.s
        rho_total[i] = (k_values[i] - 1.0) / k_values[i] * 1e5
        print(f"  k = {k_values[i]:.6f} +/- {k_std_values[i]:.6f}   ρ = {rho_total[i]:+.1f} pcm")

    rho_ref = rho_total[0]  # use the first point as reference (unrodded core)
    delta_rho = rho_total - rho_ref  # rod worth relative to unrodded
    x_crit = estimate_critical_insertion(x_arr, rho_total)

    # ── Plots ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(x_arr * 100, delta_rho, "o-", lw=2, ms=6, color="tab:blue",
             label="Δρ rod worth")
    ax1.plot(x_arr * 100, rho_total, "s--", lw=1.5, ms=4, color="tab:orange",
             label="ρ total")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("Rod insertion  [%]")
    ax1.set_ylabel("Reactivity  [pcm]")
    ax1.set_title(f"Integral reactivity  (B¹⁰ frac = {ROD_B10_FRACTION})")
    ax1.legend()

    drho = np.gradient(delta_rho, x_arr)
    ax2.plot(x_arr * 100, drho, "o-", lw=2, ms=6, color="tab:blue")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Rod insertion  [%]")
    ax2.set_ylabel("dΔρ/dx  [pcm per unit insertion]")
    ax2.set_title("Differential worth")

    plt.suptitle("OpenMC rod worth scan  (concentric geometry)", fontsize=11)
    plt.tight_layout()

    print()
    print(f"Clean-core excess reactivity: {rho_ref:+.1f} pcm")
    print(f"Full-insertion rod worth:    {delta_rho[-1]:+.1f} pcm")
    print(f"Full-insertion total ρ:      {rho_total[-1]:+.1f} pcm")
    if x_crit is not None:
        print(f"Critical insertion:          {x_crit * 100:.1f} %")
    else:
        print("Critical insertion:          not reached in [0, 1]")

    # ── Export results ───────────────────────────────────────────────────
    EXPORT_DIR = ROD_SCAN_DIR / "results"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_plots(fig, EXPORT_DIR)
    export_rod_worth_csv(x_arr, k_values, k_std_values, rho_total, delta_rho, EXPORT_DIR)
    export_run_summary_json(
        EXPORT_DIR, ROD_B10_FRACTION, ROD_DENSITY,
        SCAN_PARTICLES, SCAN_BATCHES, SCAN_INACTIVE,
        x_arr, k_values, k_std_values, rho_total, delta_rho, x_crit,
    )
    print(f"Results exported to {EXPORT_DIR.resolve()}")

else:
    print(
        "Set openmc.config['cross_sections'] or OPENMC_CROSS_SECTIONS before "
        "running rod-worth calculations."
    )

# Restore original material factory
fuel_element.make_default_materials = _original_make_materials
concentric_fuel.make_shared_materials = _original_make_materials