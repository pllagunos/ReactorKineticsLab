"""OpenMC finite-difference reactivity coefficient loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REACTIVITY_COEFFICIENTS_CSV = (
    _REPOSITORY_ROOT
    / "openmc"
    / "reference_data"
    / "concentric"
    / "reactivity_coefficients"
    / "results"
    / "reactivity_coefficients.csv"
)


@dataclass(frozen=True)
class ReactivityCoefficientSet:
    fuel_temperature_base_k: float
    fuel_temperature_pcm_per_k: float
    moderator_temperature_base_k: float
    moderator_temperature_pcm_per_k: float
    moderator_density_base_g_per_cm3: float
    moderator_density_pcm_per_g_per_cm3: float


def _required_float(row: dict[str, str], field: str, name: str) -> float:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"reactivity coefficient {name!r} missing {field}")
    return float(value)


def load_reactivity_coefficients(
    path: str | Path = DEFAULT_REACTIVITY_COEFFICIENTS_CSV,
) -> ReactivityCoefficientSet:
    table_path = Path(path)
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["name"]: row for row in csv.DictReader(handle)}

    required = {"fuel_temperature", "d2o_temperature", "d2o_density"}
    missing = required - rows.keys()
    if missing:
        raise ValueError(
            "reactivity coefficients CSV missing required rows: "
            + ", ".join(sorted(missing))
        )

    fuel = rows["fuel_temperature"]
    moderator_temperature = rows["d2o_temperature"]
    moderator_density = rows["d2o_density"]
    return ReactivityCoefficientSet(
        fuel_temperature_base_k=_required_float(
            fuel, "baseline_value", "fuel_temperature"
        ),
        fuel_temperature_pcm_per_k=_required_float(
            fuel, "coefficient_value", "fuel_temperature"
        ),
        moderator_temperature_base_k=_required_float(
            moderator_temperature, "baseline_value", "d2o_temperature"
        ),
        moderator_temperature_pcm_per_k=_required_float(
            moderator_temperature, "coefficient_value", "d2o_temperature"
        ),
        moderator_density_base_g_per_cm3=_required_float(
            moderator_density, "baseline_value", "d2o_density"
        ),
        moderator_density_pcm_per_g_per_cm3=_required_float(
            moderator_density, "coefficient_value", "d2o_density"
        ),
    )


REACTIVITY_COEFFICIENTS = load_reactivity_coefficients()
