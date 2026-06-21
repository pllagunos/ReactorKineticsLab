"""OpenMC MGXS-derived point-kinetics reference constants."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MGXS_CONSTANTS_JSON = (
    _REPOSITORY_ROOT
    / "openmc"
    / "reference_data"
    / "concentric"
    / "group_sweep"
    / "group_4"
    / "outputs"
    / "mgxs_constants.json"
)

FUEL_RING_RE = re.compile(r"^core_fuel_ring_(\d+)$")


@dataclass(frozen=True)
class KineticsDelayedGroup:
    beta: float
    decay_constant: float


@dataclass(frozen=True)
class KineticsReference:
    delayed_groups: tuple[KineticsDelayedGroup, ...]
    neutron_generation_time_seconds: float
    source_path: Path

    @property
    def beta_effective(self) -> float:
        return sum(group.beta for group in self.delayed_groups)

    @property
    def beta_effective_pcm(self) -> float:
        return self.beta_effective * 1.0e5


def _as_float_vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc


def _fuel_source_weight(domain: dict[str, Any], label: str) -> float:
    group_constants = domain.get("group_constants", {})
    nu_fission = _as_float_vector(
        group_constants.get("nu-fission", {}).get("mean"),
        f"{label} nu-fission mean",
    )
    integral_flux = _as_float_vector(
        domain.get("genfoam_aux", {}).get("integral_flux"),
        f"{label} integral flux",
    )
    if len(nu_fission) != len(integral_flux):
        raise ValueError(f"{label} nu-fission and flux vectors must have equal length")
    weight = sum(xs * flux for xs, flux in zip(nu_fission, integral_flux, strict=True))
    if weight <= 0.0:
        raise ValueError(f"{label} has non-positive fission-source weight")
    return weight


def _fuel_ring_items(domains: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[int, str, dict[str, Any]]] = []
    for label, domain in domains.items():
        match = FUEL_RING_RE.match(label)
        if match:
            if not isinstance(domain, dict):
                raise ValueError(f"{label} domain must be an object")
            items.append((int(match.group(1)), label, domain))
    if not items:
        raise ValueError("MGXS constants must contain core_fuel_ring_* domains")
    return [(label, domain) for _, label, domain in sorted(items)]


def load_kinetics_reference(
    path: str | Path = DEFAULT_MGXS_CONSTANTS_JSON,
) -> KineticsReference:
    constants_path = Path(path)
    with constants_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    run = payload.get("run", {})
    generation_time = run.get("prompt_generation_time_s")
    if generation_time is None:
        raise ValueError("MGXS constants must contain run.prompt_generation_time_s")
    generation_time = float(generation_time)
    if generation_time <= 0.0:
        raise ValueError("prompt_generation_time_s must be positive")

    domains = payload.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("MGXS constants must contain a domains object")

    weighted_beta: list[float] | None = None
    weighted_decay_numerator: list[float] | None = None
    weighted_decay_denominator: list[float] | None = None
    source_weight_total = 0.0

    for label, domain in _fuel_ring_items(domains):
        delayed = domain.get("delayed_neutrons", {})
        beta = _as_float_vector(
            delayed.get("beta_total_by_delayed_group"),
            f"{label} beta_total_by_delayed_group",
        )
        decay = _as_float_vector(
            delayed.get("decay_rate_per_s_by_delayed_group"),
            f"{label} decay_rate_per_s_by_delayed_group",
        )
        if len(beta) != len(decay):
            raise ValueError(f"{label} beta and decay vectors must have equal length")

        if weighted_beta is None:
            weighted_beta = [0.0] * len(beta)
            weighted_decay_numerator = [0.0] * len(beta)
            weighted_decay_denominator = [0.0] * len(beta)
        elif len(beta) != len(weighted_beta):
            raise ValueError("All fuel-ring delayed-neutron vectors must match")

        weight = _fuel_source_weight(domain, label)
        source_weight_total += weight
        assert weighted_decay_numerator is not None
        assert weighted_decay_denominator is not None
        for index, (beta_i, decay_i) in enumerate(zip(beta, decay, strict=True)):
            weighted_beta[index] += weight * beta_i
            beta_weight = weight * beta_i
            weighted_decay_numerator[index] += beta_weight * decay_i
            weighted_decay_denominator[index] += beta_weight

    assert weighted_beta is not None
    assert weighted_decay_numerator is not None
    assert weighted_decay_denominator is not None
    if source_weight_total <= 0.0:
        raise ValueError("Fuel-ring source weights must sum to a positive value")

    delayed_groups = tuple(
        KineticsDelayedGroup(
            beta=beta_sum / source_weight_total,
            decay_constant=(
                weighted_decay_numerator[index] / weighted_decay_denominator[index]
                if weighted_decay_denominator[index] > 0.0
                else 0.0
            ),
        )
        for index, beta_sum in enumerate(weighted_beta)
    )
    if any(group.beta < 0.0 for group in delayed_groups):
        raise ValueError("Delayed neutron beta values must be non-negative")
    if any(group.decay_constant <= 0.0 for group in delayed_groups):
        raise ValueError("Delayed neutron decay constants must be positive")

    return KineticsReference(
        delayed_groups=delayed_groups,
        neutron_generation_time_seconds=generation_time,
        source_path=constants_path,
    )


KINETICS_REFERENCE = load_kinetics_reference()
