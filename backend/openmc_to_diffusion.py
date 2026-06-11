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
from reactor_backend.multigroup_sph import (
    fit_sph_factors,
    load_sph_factors,
    qualify_mesh,
    save_sph_factors,
    validate_sph_factors,
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
    parser.add_argument(
        "--fit-sph",
        type=Path,
        metavar="PATH",
        help="Fit CE-referenced SPH factors and write them to PATH",
    )
    parser.add_argument(
        "--sph-factors",
        type=Path,
        help="Apply an existing SPH factor file when preparing the cache",
    )
    parser.add_argument(
        "--qualify-mesh",
        action="store_true",
        help=(
            "Compare the 0.5/1/5/10 cm, dz=20 cm production candidate "
            "against the reference mesh"
        ),
    )
    parser.add_argument(
        "--qualification-json",
        type=Path,
        help="Optional path for the mesh-qualification report",
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
    selected_spacing = ConcentricMeshSpacing(
        fuel_radial_cm=args.fuel_dr,
        core_coolant_radial_cm=args.core_coolant_dr,
        moderator_radial_cm=args.moderator_dr,
        reflector_radial_cm=args.reflector_dr,
        axial_cm=args.dz,
    )
    factors = None
    if args.sph_factors is not None:
        factors = load_sph_factors(args.sph_factors)
        validate_sph_factors(
            diffusion_input,
            selected_spacing,
            factors,
        )
        summary["sph_factors"] = factors.as_dict()
    if args.fit_sph is not None:
        if factors is not None:
            raise ValueError("Pass either --fit-sph or --sph-factors, not both")
        fitted = fit_sph_factors(
            diffusion_input,
            spacing=selected_spacing,
        )
        save_sph_factors(fitted.factors, args.fit_sph)
        factors = fitted.factors
        summary["sph_fit"] = {
            "factor_file": str(args.fit_sph.resolve()),
            "factors": fitted.factors.as_dict(),
            "qualification": fitted.qualification,
        }
    if args.qualify_mesh:
        qualification = qualify_mesh(
            diffusion_input,
            reference_spacing=ConcentricMeshSpacing(),
            candidate_spacing=ConcentricMeshSpacing(
                fuel_radial_cm=0.5,
                core_coolant_radial_cm=1.0,
                moderator_radial_cm=5.0,
                reflector_radial_cm=10.0,
                axial_cm=20.0,
            ),
        )
        summary["mesh_qualification"] = qualification
        if args.qualification_json is not None:
            args.qualification_json.parent.mkdir(parents=True, exist_ok=True)
            args.qualification_json.write_text(
                json.dumps(qualification, indent=2) + "\n",
                encoding="utf-8",
            )
    elif args.qualification_json is not None:
        raise ValueError("--qualification-json requires --qualify-mesh")
    if args.prepare_cache:
        settings = DiffusionCacheSettings(
            spacing=selected_spacing
        )
        prepared = prepare_concentric_diffusion_cache(
            diffusion_input,
            settings=settings,
            cache_root=args.cache_dir,
            force=args.force,
            sph_factors=factors,
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
