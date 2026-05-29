from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import openmc

from fuel_element import fuel_element_total_height_cm
from involutes import InvoluteElementParameters, build_parameter_report, validate_parameters
from ploting import resolve_openmc_exec
from reactor_geometry import ReactorTankParameters, build_reactor_model, validate_reactor_tanks


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


TARGET_KEFF = float(os.environ.get("OPT_TARGET_KEFF", "1.005"))
OPENMC_THREADS = int(os.environ.get("OPENMC_THREADS", str(os.cpu_count() or 1)))
OPENMC_EXEC = resolve_openmc_exec()
BUILD_DIR = Path(__file__).resolve().parent / "build" / "optimization"


BASE_FUEL_PARAMETERS = InvoluteElementParameters(
    plate_count=50,
    plate_thickness_cm=0.3,
    coolant_gap_cm=4.0,
    inner_radius_cm=50.0,
    outer_radius_cm=150.0,
    control_rod_radius_cm=4.0,
    h_active_cm=300.0,
    lower_plenum_cm=50.0,
    upper_plenum_cm=50.0,
    segments_per_plate=24,
    base_radius_cm=None,
    fuel_density_g_per_cm3=19.05,
    fuel_enrichment_wt_pct=0.7,
)


PLATE_COUNTS = [40, 50, 60]
PLATE_THICKNESSES_CM = [0.3, 0.5]
COOLANT_GAPS_CM = [4.0, 6.0, 8.0]
INNER_RADII_CM = [50.0, 80.0, 100.0]
OUTER_RADII_CM = [150.0, 250.0, 345.0]
ACTIVE_HEIGHTS_CM = [200.0, 300.0, 450.0]


@dataclass(frozen=True)
class SearchStage:
    name: str
    particles: int
    batches: int
    inactive: int
    keep_top_n: int
    max_cases: int | None = None


COARSE_STAGE = SearchStage(
    name="coarse",
    particles=4000,
    batches=4,
    inactive=1,
    keep_top_n=12,
    max_cases=_optional_env_int("OPT_COARSE_MAX_CASES"),
)


REFINE_STAGE = SearchStage(
    name="refine",
    particles=12000,
    batches=8,
    inactive=2,
    keep_top_n=8,
    max_cases=_optional_env_int("OPT_REFINE_MAX_CASES"),
)


@dataclass(frozen=True)
class CandidateDesign:
    plate_count: int
    plate_thickness_cm: float
    coolant_gap_cm: float
    inner_radius_cm: float
    outer_radius_cm: float
    h_active_cm: float

    @property
    def moderator_ratio(self) -> float:
        return self.coolant_gap_cm / self.plate_thickness_cm

    def key(self) -> tuple[int, float, float, float, float, float]:
        return (
            self.plate_count,
            round(self.plate_thickness_cm, 6),
            round(self.coolant_gap_cm, 6),
            round(self.inner_radius_cm, 6),
            round(self.outer_radius_cm, 6),
            round(self.h_active_cm, 6),
        )

    def label(self) -> str:
        return (
            f"pc{self.plate_count:03d}_"
            f"pt{self.plate_thickness_cm:.2f}_"
            f"cg{self.coolant_gap_cm:.2f}_"
            f"ir{self.inner_radius_cm:.1f}_"
            f"or{self.outer_radius_cm:.1f}_"
            f"ha{self.h_active_cm:.1f}"
        ).replace(".", "p")

    def fuel_parameters(self, base: InvoluteElementParameters) -> InvoluteElementParameters:
        return replace(
            base,
            plate_count=self.plate_count,
            plate_thickness_cm=self.plate_thickness_cm,
            coolant_gap_cm=self.coolant_gap_cm,
            inner_radius_cm=self.inner_radius_cm,
            outer_radius_cm=self.outer_radius_cm,
            h_active_cm=self.h_active_cm,
            base_radius_cm=None,
        )


PRIORITY_DESIGNS = [
    CandidateDesign(
        plate_count=50,
        plate_thickness_cm=0.3,
        coolant_gap_cm=4.0,
        inner_radius_cm=50.0,
        outer_radius_cm=150.0,
        h_active_cm=200.0,
    ),
    CandidateDesign(
        plate_count=50,
        plate_thickness_cm=0.3,
        coolant_gap_cm=6.0,
        inner_radius_cm=50.0,
        outer_radius_cm=150.0,
        h_active_cm=200.0,
    ),
    CandidateDesign(
        plate_count=50,
        plate_thickness_cm=0.3,
        coolant_gap_cm=8.0,
        inner_radius_cm=50.0,
        outer_radius_cm=150.0,
        h_active_cm=200.0,
    ),
    CandidateDesign(
        plate_count=50,
        plate_thickness_cm=0.3,
        coolant_gap_cm=6.0,
        inner_radius_cm=80.0,
        outer_radius_cm=345.0,
        h_active_cm=300.0,
    ),
    CandidateDesign(
        plate_count=60,
        plate_thickness_cm=0.3,
        coolant_gap_cm=8.0,
        inner_radius_cm=80.0,
        outer_radius_cm=345.0,
        h_active_cm=450.0,
    ),
    CandidateDesign(
        plate_count=60,
        plate_thickness_cm=0.3,
        coolant_gap_cm=8.0,
        inner_radius_cm=100.0,
        outer_radius_cm=345.0,
        h_active_cm=450.0,
    ),
]


@dataclass(frozen=True)
class CandidateResult:
    stage: str
    case_index: int
    label: str
    keff: float
    keff_std: float
    runtime_s: float
    plate_count: int
    plate_thickness_cm: float
    coolant_gap_cm: float
    moderator_ratio: float
    inner_radius_cm: float
    outer_radius_cm: float
    h_active_cm: float
    fuel_mass_kg: float
    uranium_area_fraction: float
    d2o_tank_radius_cm: float
    h2o_tank_radius_cm: float
    h_d2o_tank_cm: float
    h_h2o_tank_cm: float
    particles: int
    batches: int
    inactive: int
    case_dir: str

    def ranking_key(self, target_keff: float = TARGET_KEFF) -> tuple[int, float, float, float, float, float]:
        overshoot = self.keff - target_keff
        return (
            0 if overshoot >= 0.0 else 1,
            abs(overshoot),
            self.keff_std,
            -self.moderator_ratio,
            -self.outer_radius_cm,
            -self.h_active_cm,
        )

    def to_dict(self, target_keff: float = TARGET_KEFF) -> dict[str, object]:
        result = asdict(self)
        result["ranking_key"] = list(self.ranking_key(target_keff))
        result["delta_to_target"] = self.keff - target_keff
        return result


def derived_tank_parameters(fuel: InvoluteElementParameters) -> ReactorTankParameters:
    total_height_cm = fuel_element_total_height_cm(fuel)
    radial_reflector_cm = max(100.0, 1.5 * fuel.outer_radius_cm)
    light_water_shell_cm = max(100.0, 0.75 * fuel.outer_radius_cm)
    axial_reflector_cm = max(80.0, 0.30 * total_height_cm)
    light_water_axial_shell_cm = max(100.0, 0.20 * total_height_cm)

    return ReactorTankParameters(
        d2o_tank_radius_cm=fuel.outer_radius_cm + radial_reflector_cm,
        h2o_tank_radius_cm=fuel.outer_radius_cm + radial_reflector_cm + light_water_shell_cm,
        h_d2o_tank_cm=total_height_cm + 2.0 * axial_reflector_cm,
        h_h2o_tank_cm=total_height_cm + 2.0 * (axial_reflector_cm + light_water_axial_shell_cm),
    )


def candidate_is_valid(design: CandidateDesign) -> bool:
    try:
        fuel = design.fuel_parameters(BASE_FUEL_PARAMETERS)
        validate_parameters(fuel)
        validate_reactor_tanks(fuel, derived_tank_parameters(fuel))
    except ValueError:
        return False
    return True


def configure_eigenvalue_settings(
    model: openmc.Model,
    fuel: InvoluteElementParameters,
    stage: SearchStage,
) -> None:
    active_half = 0.5 * fuel.h_active_cm
    source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-fuel.outer_radius_cm, -fuel.outer_radius_cm, -active_half),
            (fuel.outer_radius_cm, fuel.outer_radius_cm, active_half),
        ),
        constraints={"fissionable": True},
    )

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = stage.particles
    settings.batches = stage.batches
    settings.inactive = stage.inactive
    settings.source = source
    settings.temperature = {"method": "interpolation"}
    model.settings = settings


def coarse_grid_candidates() -> list[CandidateDesign]:
    candidates: dict[tuple[int, float, float, float, float, float], CandidateDesign] = {}
    for candidate in PRIORITY_DESIGNS:
        if candidate_is_valid(candidate):
            candidates[candidate.key()] = candidate

    for plate_count, plate_thickness_cm, coolant_gap_cm, inner_radius_cm, outer_radius_cm, h_active_cm in product(
        PLATE_COUNTS,
        PLATE_THICKNESSES_CM,
        COOLANT_GAPS_CM,
        INNER_RADII_CM,
        OUTER_RADII_CM,
        ACTIVE_HEIGHTS_CM,
    ):
        candidate = CandidateDesign(
            plate_count=plate_count,
            plate_thickness_cm=plate_thickness_cm,
            coolant_gap_cm=coolant_gap_cm,
            inner_radius_cm=inner_radius_cm,
            outer_radius_cm=outer_radius_cm,
            h_active_cm=h_active_cm,
        )
        if candidate_is_valid(candidate):
            candidates[candidate.key()] = candidate

    prioritized_keys = {candidate.key() for candidate in PRIORITY_DESIGNS if candidate_is_valid(candidate)}
    ordered_candidates = list(candidates.values())
    ordered_candidates.sort(
        key=lambda candidate: (
            0 if candidate.key() in prioritized_keys else 1,
            -candidate.moderator_ratio,
            -(candidate.outer_radius_cm - candidate.inner_radius_cm),
            -candidate.outer_radius_cm,
            -candidate.h_active_cm,
            -candidate.plate_count,
            candidate.plate_thickness_cm,
        )
    )
    return ordered_candidates


def refined_candidates(seed_results: list[CandidateResult]) -> list[CandidateDesign]:
    deltas = {
        "plate_count": (-10, -4, 4, 10),
        "plate_thickness_cm": (-0.10, 0.10),
        "coolant_gap_cm": (-1.0, 1.0, 2.0),
        "inner_radius_cm": (-10.0, 10.0),
        "outer_radius_cm": (-25.0, 25.0),
        "h_active_cm": (-100.0, 100.0),
    }
    candidates: dict[tuple[int, float, float, float, float, float], CandidateDesign] = {}

    for result in seed_results:
        seed = CandidateDesign(
            plate_count=result.plate_count,
            plate_thickness_cm=result.plate_thickness_cm,
            coolant_gap_cm=result.coolant_gap_cm,
            inner_radius_cm=result.inner_radius_cm,
            outer_radius_cm=result.outer_radius_cm,
            h_active_cm=result.h_active_cm,
        )
        candidates[seed.key()] = seed

        for field_name, field_deltas in deltas.items():
            for delta in field_deltas:
                updates = {
                    "plate_count": seed.plate_count,
                    "plate_thickness_cm": seed.plate_thickness_cm,
                    "coolant_gap_cm": seed.coolant_gap_cm,
                    "inner_radius_cm": seed.inner_radius_cm,
                    "outer_radius_cm": seed.outer_radius_cm,
                    "h_active_cm": seed.h_active_cm,
                }
                updates[field_name] = updates[field_name] + delta
                if field_name == "plate_count":
                    updates[field_name] = max(8, int(round(updates[field_name] / 2.0) * 2))
                else:
                    updates[field_name] = round(float(updates[field_name]), 3)

                if updates["plate_thickness_cm"] <= 0.0 or updates["coolant_gap_cm"] <= 0.0:
                    continue
                if updates["inner_radius_cm"] <= 0.0 or updates["outer_radius_cm"] <= updates["inner_radius_cm"]:
                    continue
                if updates["h_active_cm"] <= 0.0:
                    continue

                candidate = CandidateDesign(**updates)
                if candidate_is_valid(candidate):
                    candidates[candidate.key()] = candidate

    refined = list(candidates.values())
    refined.sort(
        key=lambda candidate: (
            -candidate.moderator_ratio,
            -candidate.outer_radius_cm,
            -candidate.h_active_cm,
            -candidate.plate_count,
            candidate.plate_thickness_cm,
        )
    )
    return refined


def evaluate_candidate(
    design: CandidateDesign,
    stage: SearchStage,
    case_index: int,
) -> CandidateResult:
    fuel = design.fuel_parameters(BASE_FUEL_PARAMETERS)
    validate_parameters(fuel)
    tanks = derived_tank_parameters(fuel)
    validate_reactor_tanks(fuel, tanks)

    report = build_parameter_report(fuel)
    model, _ = build_reactor_model(fuel, tanks, rod_insertion=0.0)
    configure_eigenvalue_settings(model, fuel, stage)

    case_dir = BUILD_DIR / stage.name / f"{case_index:04d}_{design.label()}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    with (case_dir / "inputs.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fuel_parameters": fuel.to_geometry_dict(),
                "tank_parameters": tanks.to_geometry_dict(),
                "stage": asdict(stage),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    started_at = time.perf_counter()
    statepoint_path = model.run(
        cwd=case_dir,
        threads=OPENMC_THREADS,
        openmc_exec=OPENMC_EXEC,
        export_model_xml=True,
        output=False,
    )
    runtime_s = time.perf_counter() - started_at

    with openmc.StatePoint(statepoint_path) as statepoint:
        keff = statepoint.keff

    result = CandidateResult(
        stage=stage.name,
        case_index=case_index,
        label=design.label(),
        keff=keff.n,
        keff_std=keff.s,
        runtime_s=runtime_s,
        plate_count=design.plate_count,
        plate_thickness_cm=design.plate_thickness_cm,
        coolant_gap_cm=design.coolant_gap_cm,
        moderator_ratio=design.moderator_ratio,
        inner_radius_cm=design.inner_radius_cm,
        outer_radius_cm=design.outer_radius_cm,
        h_active_cm=design.h_active_cm,
        fuel_mass_kg=float(report["uranium_mass_kg"]),
        uranium_area_fraction=float(report["uranium_area_fraction"]),
        d2o_tank_radius_cm=tanks.d2o_tank_radius_cm,
        h2o_tank_radius_cm=tanks.h2o_tank_radius_cm,
        h_d2o_tank_cm=tanks.h_d2o_tank_cm,
        h_h2o_tank_cm=tanks.h_h2o_tank_cm,
        particles=stage.particles,
        batches=stage.batches,
        inactive=stage.inactive,
        case_dir=str(case_dir),
    )

    with (case_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, sort_keys=True)

    return result


def run_stage(candidates: list[CandidateDesign], stage: SearchStage) -> list[CandidateResult]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    stage_dir = BUILD_DIR / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage.max_cases is not None:
        candidates = candidates[: stage.max_cases]

    results: list[CandidateResult] = []
    failures: list[dict[str, str]] = []

    for index, candidate in enumerate(candidates, start=1):
        print(f"[{stage.name}] {index}/{len(candidates)} {candidate.label()}")
        try:
            result = evaluate_candidate(candidate, stage, index)
        except ValueError as exc:
            failures.append({"label": candidate.label(), "error": str(exc)})
            print(f"  skipped invalid design: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - depends on local OpenMC runtime state
            failures.append({"label": candidate.label(), "error": str(exc)})
            print(f"  OpenMC failed: {exc}")
            continue

        results.append(result)
        print(
            "  "
            f"k_eff = {result.keff:.5f} +/- {result.keff_std:.5f}; "
            f"fuel mass = {result.fuel_mass_kg:.1f} kg; "
            f"runtime = {result.runtime_s:.1f} s"
        )

    results.sort(key=lambda result: result.ranking_key())

    with (stage_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump([result.to_dict() for result in results], handle, indent=2, sort_keys=True)

    with (stage_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(results[0].to_dict().keys()) if results else ["label", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if results:
            for result in results:
                writer.writerow(result.to_dict())
        else:
            for failure in failures:
                writer.writerow(failure)

    if failures:
        with (stage_dir / "failures.json").open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2, sort_keys=True)

    return results


def print_top_results(results: list[CandidateResult], limit: int = 8) -> None:
    if not results:
        print("No valid OpenMC results were produced.")
        return

    print("\nTop candidates nearest the target k_eff:")
    for rank, result in enumerate(results[:limit], start=1):
        delta = result.keff - TARGET_KEFF
        print(
            f"{rank:02d}. {result.label}: "
            f"k_eff = {result.keff:.5f} +/- {result.keff_std:.5f}, "
            f"delta = {delta:+.5f}, "
            f"gap/thickness = {result.moderator_ratio:.2f}, "
            f"outer radius = {result.outer_radius_cm:.1f} cm, "
            f"active height = {result.h_active_cm:.1f} cm, "
            f"fuel mass = {result.fuel_mass_kg:.1f} kg"
        )


def write_final_summary(
    coarse_results: list[CandidateResult],
    refined_results: list[CandidateResult],
) -> None:
    best_results = refined_results if refined_results else coarse_results
    summary = {
        "target_keff": TARGET_KEFF,
        "openmc_threads": OPENMC_THREADS,
        "openmc_exec": OPENMC_EXEC,
        "base_fuel_parameters": BASE_FUEL_PARAMETERS.to_geometry_dict(),
        "coarse_stage": asdict(COARSE_STAGE),
        "refine_stage": asdict(REFINE_STAGE),
        "coarse_results_count": len(coarse_results),
        "refined_results_count": len(refined_results),
        "best_results": [result.to_dict() for result in best_results[:10]],
    }
    with (BUILD_DIR / "search_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def run_search() -> tuple[list[CandidateResult], list[CandidateResult]]:
    coarse_candidates = coarse_grid_candidates()
    print(
        f"Running coarse search across {len(coarse_candidates)} grid points "
        f"with target k_eff = {TARGET_KEFF:.5f}."
    )
    coarse_results = run_stage(coarse_candidates, COARSE_STAGE)

    seed_results = coarse_results[: COARSE_STAGE.keep_top_n]
    refine_candidates = refined_candidates(seed_results)
    print(
        f"\nRefining around {len(seed_results)} coarse candidates with "
        f"{len(refine_candidates)} local perturbations."
    )
    refined_results = run_stage(refine_candidates, REFINE_STAGE)

    write_final_summary(coarse_results, refined_results)
    return coarse_results, refined_results


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    coarse_results, refined_results = run_search()
    best_results = refined_results if refined_results else coarse_results
    print_top_results(best_results)
    print(f"\nSearch artifacts written to {BUILD_DIR}")


if __name__ == "__main__":
    main()