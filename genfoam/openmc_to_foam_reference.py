from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from openmc_to_genfoam_xs import DEFAULT_OUTPUT_SUBDIR, generate_genfoam_xs
from prepare_concentric_case import DEFAULT_MGXS_EXPORT_DIR, DEFAULT_OUTPUT_DIR

DEFAULT_REFERENCE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "mgxs_to_genfoam_reference"

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


def run_comparison(
    mgxs_export_dir: Path,
    output_dir: Path,
    rerun_mgxs: bool,
    particles: int | None,
    batches: int | None,
    inactive: int | None,
    legendre_order: int | None,
    threads: int | None,
) -> dict[str, Any]:
    xs_output_dir = output_dir / DEFAULT_OUTPUT_SUBDIR
    xs_payload = generate_genfoam_xs(
        mgxs_export_dir=mgxs_export_dir,
        output_dir=xs_output_dir,
        rerun_mgxs=rerun_mgxs,
        particles=particles,
        batches=batches,
        inactive=inactive,
        legendre_order=legendre_order,
        threads=threads,
    )

    comparison: dict[str, Any] = {
        "zones": {},
        "notes": list(xs_payload.get("notes", [])),
    }
    for zone_name, zone_reference in xs_payload["raw_zones"].items():
        zone_current = xs_payload["zones"][zone_name]
        comparison["zones"][zone_name] = {
            field_name: _compare_vectors(zone_reference[field_name], zone_current[field_name])
            for field_name in ("IV", "D", "nuSigmaEff", "sigmaPow", "sigmaRemoval", "chiPrompt", "chiDelayed", "Beta", "lambda", "integralFlux")
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.json"
    summary_path = output_dir / "summary.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    summary = {
        "mgxs_export_dir": str(mgxs_export_dir),
        "reference_run": xs_payload["reference_run"],
        "xs_summary_path": xs_payload["files"]["summary"],
        "raw_vectors_path": xs_payload["files"]["raw_vectors"],
        "adapted_vectors_path": xs_payload["files"]["adapted_vectors"],
        "comparison_path": str(comparison_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw MGXS-export-derived zone vectors with the sanitized GeN-Foam writer outputs."
    )
    parser.add_argument("--mgxs-export-dir", type=Path, default=DEFAULT_MGXS_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REFERENCE_OUTPUT_DIR)
    parser.add_argument("--rerun-mgxs", action="store_true")
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--inactive", type=int, default=None)
    parser.add_argument("--legendre-order", type=int, default=None)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_comparison(
        mgxs_export_dir=args.mgxs_export_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        rerun_mgxs=args.rerun_mgxs,
        particles=args.particles,
        batches=args.batches,
        inactive=args.inactive,
        legendre_order=args.legendre_order,
        threads=args.threads,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
