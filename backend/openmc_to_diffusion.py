#!/usr/bin/env python3
"""Validate and summarize an OpenMC MGXS export for the diffusion solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactor_backend.multigroup_diffusion import ConcentricMeshSpacing
from reactor_backend.multigroup_diffusion_cache import (
    DEFAULT_CACHE_ROOT,
    DiffusionCacheSettings,
    prepare_concentric_diffusion_cache,
)
from reactor_backend.openmc_mgxs_adapter import (
    load_concentric_diffusion_input,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a resolved OpenMC MGXS export, enforce the diffusion "
            "scattering contract, and report the normalized solver input."
        )
    )
    parser.add_argument(
        "--mgxs-export-dir",
        type=Path,
        required=True,
        help=(
            "Export directory containing reactor_run/model.xml and "
            "outputs/mgxs_constants.json"
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path for the normalized validation summary",
    )
    parser.add_argument(
        "--prepare-cache",
        action="store_true",
        help="Build or load the persistent group matrices and clean solution",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Cache root (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the matching cache entry",
    )
    parser.add_argument("--fuel-dr", type=float, default=0.1)
    parser.add_argument("--core-coolant-dr", type=float, default=1.0)
    parser.add_argument("--moderator-dr", type=float, default=5.0)
    parser.add_argument("--reflector-dr", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    diffusion_input = load_concentric_diffusion_input(args.mgxs_export_dir)
    summary = diffusion_input.summary()
    if args.prepare_cache:
        settings = DiffusionCacheSettings(
            spacing=ConcentricMeshSpacing(
                fuel_radial_cm=args.fuel_dr,
                core_coolant_radial_cm=args.core_coolant_dr,
                moderator_radial_cm=args.moderator_dr,
                reflector_radial_cm=args.reflector_dr,
                axial_cm=args.dz,
            )
        )
        prepared = prepare_concentric_diffusion_cache(
            diffusion_input,
            settings=settings,
            cache_root=args.cache_dir,
            force=args.force,
        )
        summary["cache"] = prepared.summary()
    elif args.force:
        raise ValueError("--force requires --prepare-cache")
    output = json.dumps(summary, indent=2) + "\n"
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
