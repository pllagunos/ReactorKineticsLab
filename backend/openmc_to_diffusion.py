#!/usr/bin/env python3
"""Validate and summarize an OpenMC MGXS export for the diffusion solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser


def main() -> int:
    args = _parser().parse_args()
    diffusion_input = load_concentric_diffusion_input(args.mgxs_export_dir)
    summary = diffusion_input.summary()
    output = json.dumps(summary, indent=2) + "\n"
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
