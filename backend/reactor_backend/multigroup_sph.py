"""Superhomogenization support for resolved multigroup diffusion models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .multigroup_diffusion import (
    BOUNDARY_CONDITIONS,
    BOUNDARY_EXTRAPOLATED_MESH,
    ConcentricMeshSpacing,
    MultiGroupDiffusionSystem,
    MultiGroupRegion,
    _power_density,
    build_concentric_mesh,
    build_multigroup_2d_system,
    solve_multigroup_system,
)
from .openmc_mgxs_adapter import ConcentricDiffusionInput


SPH_ALGORITHM_VERSION = 4
REFERENCE_MODE_REGION_FLUX = "region_flux"
REFERENCE_MODE_AXIAL_POWER_SHAPE = "axial_power_shape"
REFERENCE_MODE_AXIAL_REGION_FLUX = "axial_region_flux"
SPH_REFERENCE_MODES = (
    REFERENCE_MODE_REGION_FLUX,
    REFERENCE_MODE_AXIAL_POWER_SHAPE,
    REFERENCE_MODE_AXIAL_REGION_FLUX,
)
AXIAL_SPH_LABEL_SEPARATOR = "__axial_"

AxialSphZones = Mapping[str, tuple[tuple[str, float, float], ...]]


def _normalize_reference_mode(reference_mode: str) -> str:
    mode = str(reference_mode)
    if mode not in SPH_REFERENCE_MODES:
        raise ValueError(
            "SPH reference_mode must be one of "
            f"{', '.join(SPH_REFERENCE_MODES)}"
        )
    return mode


def _normalize_axial_sph_zones(
    axial_sph_zones: Mapping[str, Any] | None,
) -> Mapping[str, tuple[tuple[str, float, float], ...]]:
    if axial_sph_zones is None:
        return MappingProxyType({})
    normalized: dict[str, tuple[tuple[str, float, float], ...]] = {}
    for raw_label, raw_entries in axial_sph_zones.items():
        label = str(raw_label)
        if not label:
            raise ValueError("AXIAL_SPH_ZONES labels must be non-empty")
        entries = []
        seen_names: set[str] = set()
        for raw_entry in raw_entries:
            if len(raw_entry) != 3:
                raise ValueError(
                    "Each AXIAL_SPH_ZONES entry must be "
                    "(zone_name, z_min_cm, z_max_cm)"
                )
            zone_name = str(raw_entry[0])
            if (
                not zone_name
                or AXIAL_SPH_LABEL_SEPARATOR in zone_name
                or any(character.isspace() for character in zone_name)
            ):
                raise ValueError(
                    "Axial SPH zone names must be non-empty labels without "
                    "whitespace or the reserved separator"
                )
            if zone_name in seen_names:
                raise ValueError(
                    f"Duplicate axial SPH zone name {zone_name!r} "
                    f"for region {label!r}"
                )
            seen_names.add(zone_name)
            z_min = float(raw_entry[1])
            z_max = float(raw_entry[2])
            if (
                not math.isfinite(z_min)
                or not math.isfinite(z_max)
                or z_max <= z_min
            ):
                raise ValueError(
                    "Axial SPH zone z bounds must be finite and increasing"
                )
            entries.append((zone_name, z_min, z_max))
        if not entries:
            raise ValueError(
                f"AXIAL_SPH_ZONES region {label!r} has no axial zones"
            )
        entries.sort(key=lambda item: (item[1], item[2], item[0]))
        normalized[label] = tuple(entries)
    return MappingProxyType(normalized)


def _axial_sph_zones_as_dict(
    axial_sph_zones: Mapping[str, tuple[tuple[str, float, float], ...]],
) -> dict[str, list[list[str | float]]]:
    return {
        label: [
            [zone_name, float(z_min), float(z_max)]
            for zone_name, z_min, z_max in entries
        ]
        for label, entries in axial_sph_zones.items()
    }


def _axial_region_label(region_label: str, zone_name: str) -> str:
    return f"{region_label}{AXIAL_SPH_LABEL_SEPARATOR}{zone_name}"


def _base_region_label(region_label: str) -> str:
    return region_label.split(AXIAL_SPH_LABEL_SEPARATOR, 1)[0]


def _readonly_array(
    values: Any,
    *,
    dtype: Any = float,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SphSettings:
    damping: float = 0.5
    max_iterations: int = 50
    flux_tolerance: float = 2.0e-3
    factor_stability_tolerance: float = 2.0e-3
    k_stability_pcm: float = 5.0
    stable_iterations: int = 2
    minimum_factor: float = 0.1
    maximum_factor: float = 10.0
    active_flux_fraction: float = 1.0e-10
    maximum_relative_std_dev: float = 0.5
    excluded_region_labels: tuple[str, ...] = (
        "reflector",
        "core_control_rod",
    )
    qualification_flux_tolerance: float = 1.0e-2
    qualification_power_rms_tolerance: float | None = None
    qualification_radial_power_rms_tolerance: float = 8.0e-2
    qualification_axial_power_rms_tolerance: float = 1.0e-1
    boundary_condition: str = BOUNDARY_EXTRAPOLATED_MESH
    solver_max_iterations: int = 300
    solver_tolerance: float = 1.0e-6
    solver_source_tolerance: float = 1.0e-3
    solver_max_inner_iterations: int = 200
    solver_inner_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("SPH damping must lie in (0, 1]")
        if self.max_iterations < 1 or self.stable_iterations < 1:
            raise ValueError("SPH iteration limits must be positive")
        if (
            self.flux_tolerance <= 0.0
            or self.factor_stability_tolerance <= 0.0
            or self.k_stability_pcm <= 0.0
        ):
            raise ValueError("SPH convergence tolerances must be positive")
        if not 0.0 < self.minimum_factor < self.maximum_factor:
            raise ValueError("SPH factor bounds are invalid")
        if self.active_flux_fraction < 0.0:
            raise ValueError("SPH active-flux threshold must be non-negative")
        if self.maximum_relative_std_dev <= 0.0:
            raise ValueError("SPH relative uncertainty limit must be positive")
        if self.qualification_flux_tolerance <= 0.0:
            raise ValueError("SPH flux qualification tolerance must be positive")
        if (
            self.qualification_power_rms_tolerance is not None
            and self.qualification_power_rms_tolerance <= 0.0
        ):
            raise ValueError("SPH power RMS tolerance must be positive")
        if (
            self.qualification_radial_power_rms_tolerance <= 0.0
            or self.qualification_axial_power_rms_tolerance <= 0.0
        ):
            raise ValueError("SPH power RMS tolerances must be positive")
        if self.boundary_condition not in BOUNDARY_CONDITIONS:
            raise ValueError(
                "SPH boundary_condition must be one of "
                f"{', '.join(BOUNDARY_CONDITIONS)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @property
    def radial_power_rms_tolerance(self) -> float:
        return (
            self.qualification_radial_power_rms_tolerance
            if self.qualification_power_rms_tolerance is None
            else self.qualification_power_rms_tolerance
        )

    @property
    def axial_power_rms_tolerance(self) -> float:
        return (
            self.qualification_axial_power_rms_tolerance
            if self.qualification_power_rms_tolerance is None
            else self.qualification_power_rms_tolerance
        )


@dataclass(frozen=True)
class SphFactorSet:
    region_labels: tuple[str, ...]
    factors: np.ndarray
    active: np.ndarray
    converged: bool
    iterations: int
    history: tuple[Mapping[str, float | int], ...]
    source_fingerprint: str
    mesh_spacing: Mapping[str, float]
    provisional: bool
    reference_mode: str = REFERENCE_MODE_REGION_FLUX
    axial_sph_zones: Mapping[str, tuple[tuple[str, float, float], ...]] | None = None
    boundary_condition: str = BOUNDARY_EXTRAPOLATED_MESH
    algorithm_version: int = SPH_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        factors = _readonly_array(self.factors)
        active = _readonly_array(self.active, dtype=bool)
        reference_mode = _normalize_reference_mode(self.reference_mode)
        axial_sph_zones = _normalize_axial_sph_zones(self.axial_sph_zones)
        if reference_mode in (
            REFERENCE_MODE_AXIAL_POWER_SHAPE,
            REFERENCE_MODE_AXIAL_REGION_FLUX,
        ) and not axial_sph_zones:
            raise ValueError(
                "Axial SPH factors require axial_sph_zones"
            )
        if reference_mode == REFERENCE_MODE_REGION_FLUX and axial_sph_zones:
            raise ValueError(
                "Region-flux SPH factors must not declare axial_sph_zones"
            )
        if factors.ndim != 2:
            raise ValueError("SPH factors must have shape (regions, groups)")
        if active.shape != factors.shape:
            raise ValueError("SPH active mask must match factor shape")
        if factors.shape[0] != len(self.region_labels):
            raise ValueError("SPH region labels do not match factor rows")
        if (
            not np.all(np.isfinite(factors))
            or np.any(factors <= 0.0)
        ):
            raise ValueError("SPH factors must be finite and positive")
        if self.iterations < 0:
            raise ValueError("SPH iteration count cannot be negative")
        if self.boundary_condition not in BOUNDARY_CONDITIONS:
            raise ValueError(
                "SPH factor boundary_condition must be one of "
                f"{', '.join(BOUNDARY_CONDITIONS)}"
            )
        history = tuple(
            MappingProxyType(dict(item))
            for item in self.history
        )
        mesh_spacing = MappingProxyType(
            {
                str(key): float(value)
                for key, value in self.mesh_spacing.items()
            }
        )
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "mesh_spacing", mesh_spacing)
        object.__setattr__(self, "reference_mode", reference_mode)
        object.__setattr__(self, "axial_sph_zones", axial_sph_zones)

    @property
    def group_count(self) -> int:
        return self.factors.shape[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm_version": self.algorithm_version,
            "region_labels": list(self.region_labels),
            "factors": self.factors.tolist(),
            "active": self.active.tolist(),
            "converged": self.converged,
            "iterations": self.iterations,
            "history": [dict(item) for item in self.history],
            "source_fingerprint": self.source_fingerprint,
            "mesh_spacing": dict(self.mesh_spacing),
            "provisional": self.provisional,
            "reference_mode": self.reference_mode,
            "axial_sph_zones": _axial_sph_zones_as_dict(
                self.axial_sph_zones
            ),
            "boundary_condition": self.boundary_condition,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SphFactorSet":
        return cls(
            algorithm_version=int(payload["algorithm_version"]),
            region_labels=tuple(str(value) for value in payload["region_labels"]),
            factors=np.asarray(payload["factors"], dtype=float),
            active=np.asarray(payload["active"], dtype=bool),
            converged=bool(payload["converged"]),
            iterations=int(payload["iterations"]),
            history=tuple(dict(item) for item in payload.get("history", [])),
            source_fingerprint=str(payload["source_fingerprint"]),
            mesh_spacing={
                str(key): float(value)
                for key, value in payload["mesh_spacing"].items()
            },
            provisional=bool(payload.get("provisional", True)),
            reference_mode=str(
                payload.get("reference_mode", REFERENCE_MODE_REGION_FLUX)
            ),
            axial_sph_zones=payload.get("axial_sph_zones", {}),
            boundary_condition=str(
                payload.get("boundary_condition", BOUNDARY_EXTRAPOLATED_MESH)
            ),
        )


@dataclass(frozen=True)
class SphFitResult:
    factors: SphFactorSet
    system: MultiGroupDiffusionSystem
    solution: dict[str, Any]
    qualification: dict[str, Any]
    factor_history: tuple[np.ndarray, ...]


class SphConvergenceError(RuntimeError):
    def __init__(self, message: str, result: SphFitResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class SphReferenceData:
    region_labels: tuple[str, ...]
    flux: np.ndarray
    std_dev: np.ndarray
    relative_std_dev: np.ndarray
    active: np.ndarray
    fission_production: float

    def __post_init__(self) -> None:
        flux = _readonly_array(self.flux)
        std_dev = _readonly_array(self.std_dev)
        relative_std_dev = _readonly_array(self.relative_std_dev)
        active = _readonly_array(self.active, dtype=bool)
        if flux.ndim != 2:
            raise ValueError("SPH reference flux must have shape (regions, groups)")
        expected = (len(self.region_labels), flux.shape[1])
        if flux.shape != expected:
            raise ValueError("SPH reference flux must have shape (regions, groups)")
        if (
            std_dev.shape != flux.shape
            or relative_std_dev.shape != flux.shape
            or active.shape != flux.shape
        ):
            raise ValueError("SPH reference arrays must have matching shapes")
        if (
            not math.isfinite(self.fission_production)
            or self.fission_production <= 0.0
        ):
            raise ValueError("SPH reference fission production must be positive")
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "std_dev", std_dev)
        object.__setattr__(self, "relative_std_dev", relative_std_dev)
        object.__setattr__(self, "active", active)


@dataclass(frozen=True)
class SphPowerReferenceData:
    region_labels: tuple[str, ...]
    power: np.ndarray
    std_dev: np.ndarray
    relative_std_dev: np.ndarray
    active: np.ndarray
    total_power: float

    def __post_init__(self) -> None:
        power = _readonly_array(self.power)
        std_dev = _readonly_array(self.std_dev)
        relative_std_dev = _readonly_array(self.relative_std_dev)
        active = _readonly_array(self.active, dtype=bool)
        expected = (len(self.region_labels),)
        if power.shape != expected:
            raise ValueError("SPH power reference must have shape (regions,)")
        if (
            std_dev.shape != power.shape
            or relative_std_dev.shape != power.shape
            or active.shape != power.shape
        ):
            raise ValueError("SPH power reference arrays must have matching shapes")
        if (
            not math.isfinite(self.total_power)
            or self.total_power <= 0.0
        ):
            raise ValueError("SPH power reference total must be positive")
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "std_dev", std_dev)
        object.__setattr__(self, "relative_std_dev", relative_std_dev)
        object.__setattr__(self, "active", active)


def _validate_axial_zone_coverage(
    diffusion_input: ConcentricDiffusionInput,
    axial_sph_zones: Mapping[str, tuple[tuple[str, float, float], ...]],
) -> None:
    tolerance = 1.0e-9
    unknown = set(axial_sph_zones) - set(diffusion_input.regions)
    if unknown:
        raise ValueError(f"Unknown axial SPH regions: {sorted(unknown)}")

    for label, entries in axial_sph_zones.items():
        source_zones = [
            zone for zone in diffusion_input.zones if zone.region.name == label
        ]
        if not source_zones:
            raise ValueError(f"Axial SPH region {label!r} has no diffusion zone")

        for zone_name, z_min, z_max in entries:
            if not any(
                min(z_max, source_zone.z_max) - max(z_min, source_zone.z_min)
                > tolerance
                for source_zone in source_zones
            ):
                raise ValueError(
                    f"Axial SPH zone {zone_name!r} for {label!r} does not "
                    "overlap the source diffusion zone"
                )

        for source_zone in source_zones:
            overlaps: list[tuple[float, float, str]] = []
            for zone_name, z_min, z_max in entries:
                overlap_min = max(z_min, source_zone.z_min)
                overlap_max = min(z_max, source_zone.z_max)
                if overlap_max - overlap_min > tolerance:
                    overlaps.append((overlap_min, overlap_max, zone_name))
            overlaps.sort(key=lambda item: (item[0], item[1], item[2]))
            cursor = source_zone.z_min
            for overlap_min, overlap_max, zone_name in overlaps:
                if overlap_min > cursor + tolerance:
                    raise ValueError(
                        f"Axial SPH zones for {label!r} leave a gap before "
                        f"{zone_name!r}"
                    )
                if overlap_min < cursor - tolerance:
                    raise ValueError(
                        f"Axial SPH zones for {label!r} overlap near "
                        f"{zone_name!r}"
                    )
                cursor = max(cursor, overlap_max)
            if cursor < source_zone.z_max - tolerance:
                raise ValueError(
                    f"Axial SPH zones for {label!r} do not cover the full "
                    "source diffusion zone"
                )


def axialized_diffusion_input(
    diffusion_input: ConcentricDiffusionInput,
    axial_sph_zones: Mapping[str, Any] | None,
) -> ConcentricDiffusionInput:
    """Return a diffusion input whose selected regions are split axially."""
    normalized_zones = _normalize_axial_sph_zones(axial_sph_zones)
    if not normalized_zones:
        raise ValueError("Axial SPH requires at least one region split")
    _validate_axial_zone_coverage(diffusion_input, normalized_zones)

    selected = set(normalized_zones)
    new_regions: dict[str, MultiGroupRegion] = {}
    for label, region in diffusion_input.regions.items():
        if label not in selected:
            new_regions[label] = region
            continue
        for zone_name, _, _ in normalized_zones[label]:
            clone_label = _axial_region_label(label, zone_name)
            if clone_label in diffusion_input.regions or clone_label in new_regions:
                raise ValueError(f"Duplicate axial SPH clone label {clone_label!r}")
            new_regions[clone_label] = replace(region, name=clone_label)

    new_zones = []
    new_report: list[dict[str, Any]] = []
    for source_zone, source_report in zip(
        diffusion_input.zones,
        diffusion_input.zone_report,
        strict=True,
    ):
        label = source_zone.region.name
        if label not in selected:
            new_zones.append(replace(source_zone, region=new_regions[label]))
            new_report.append(dict(source_report))
            continue
        for zone_name, z_min, z_max in normalized_zones[label]:
            overlap_min = max(z_min, source_zone.z_min)
            overlap_max = min(z_max, source_zone.z_max)
            if overlap_max <= overlap_min:
                continue
            clone_label = _axial_region_label(label, zone_name)
            new_zones.append(
                replace(
                    source_zone,
                    region=new_regions[clone_label],
                    z_min=overlap_min,
                    z_max=overlap_max,
                )
            )
            report = dict(source_report)
            report.update(
                {
                    "label": clone_label,
                    "base_label": label,
                    "axial_zone_name": zone_name,
                    "z_min_cm": overlap_min,
                    "z_max_cm": overlap_max,
                }
            )
            new_report.append(report)

    new_fuel_labels: list[str] = []
    for label in diffusion_input.fuel_ring_labels:
        if label in selected:
            new_fuel_labels.extend(
                _axial_region_label(label, zone_name)
                for zone_name, _, _ in normalized_zones[label]
            )
        else:
            new_fuel_labels.append(label)

    new_mapping = dict(diffusion_input.domain_mapping)
    for label, entries in normalized_zones.items():
        for zone_name, _, _ in entries:
            new_mapping[_axial_region_label(label, zone_name)] = (
                diffusion_input.domain_mapping[label]
            )

    return replace(
        diffusion_input,
        regions=new_regions,
        zones=tuple(new_zones),
        zone_report=tuple(new_report),
        fuel_ring_labels=tuple(new_fuel_labels),
        domain_mapping=new_mapping,
    )


def _sph_diffusion_input(
    diffusion_input: ConcentricDiffusionInput,
    reference_mode: str,
    axial_sph_zones: Mapping[str, Any] | None,
) -> ConcentricDiffusionInput:
    mode = _normalize_reference_mode(reference_mode)
    if mode == REFERENCE_MODE_REGION_FLUX:
        if _normalize_axial_sph_zones(axial_sph_zones):
            raise ValueError("Region-flux SPH does not accept axial_sph_zones")
        return diffusion_input
    return axialized_diffusion_input(diffusion_input, axial_sph_zones)


def sph_diffusion_input(
    diffusion_input: ConcentricDiffusionInput,
    factor_set: SphFactorSet,
) -> ConcentricDiffusionInput:
    return _sph_diffusion_input(
        diffusion_input,
        factor_set.reference_mode,
        factor_set.axial_sph_zones,
    )


def save_sph_factors(
    factor_set: SphFactorSet,
    path: str | Path,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(factor_set.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_sph_factors(path: str | Path) -> SphFactorSet:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SPH factor file must contain a JSON object")
    factors = SphFactorSet.from_dict(payload)
    if factors.algorithm_version != SPH_ALGORITHM_VERSION:
        raise ValueError(
            "SPH factor algorithm version does not match this backend"
        )
    return factors


def sph_source_fingerprint(
    diffusion_input: ConcentricDiffusionInput,
    spacing: ConcentricMeshSpacing,
    boundary_condition: str = BOUNDARY_EXTRAPOLATED_MESH,
    *,
    reference_mode: str = REFERENCE_MODE_REGION_FLUX,
    axial_sph_zones: Mapping[str, Any] | None = None,
) -> str:
    if boundary_condition not in BOUNDARY_CONDITIONS:
        raise ValueError(
            "boundary_condition must be one of "
            f"{', '.join(BOUNDARY_CONDITIONS)}"
        )
    mode = _normalize_reference_mode(reference_mode)
    normalized_zones = _normalize_axial_sph_zones(axial_sph_zones)
    if mode in (
        REFERENCE_MODE_AXIAL_POWER_SHAPE,
        REFERENCE_MODE_AXIAL_REGION_FLUX,
    ) and not normalized_zones:
        raise ValueError("Axial SPH requires axial_sph_zones")
    if mode == REFERENCE_MODE_REGION_FLUX and normalized_zones:
        raise ValueError("Region-flux SPH does not accept axial_sph_zones")
    digest = hashlib.sha256()
    digest.update(f"sph:{SPH_ALGORITHM_VERSION}\n".encode())
    for path in (
        diffusion_input.mgxs_json_path,
        diffusion_input.model_xml_path,
    ):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(
        json.dumps(
            spacing.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(boundary_condition.encode())
    digest.update(mode.encode())
    digest.update(
        json.dumps(
            _axial_sph_zones_as_dict(normalized_zones),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    if mode == REFERENCE_MODE_AXIAL_REGION_FLUX:
        reference = diffusion_input.ce_reference
        axial_region_flux = (
            {} if reference is None else reference.axial_region_flux
        )
        digest.update(
            json.dumps(
                {
                    label: {
                        "mean": values.mean.tolist(),
                        "std_dev": values.std_dev.tolist(),
                    }
                    for label, values in sorted(axial_region_flux.items())
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def _ordered_region_labels(
    diffusion_input: ConcentricDiffusionInput,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(zone.region.name for zone in diffusion_input.zones)
    )


def validate_sph_factors(
    diffusion_input: ConcentricDiffusionInput,
    spacing: ConcentricMeshSpacing,
    factor_set: SphFactorSet,
) -> None:
    adjusted_input = sph_diffusion_input(diffusion_input, factor_set)
    expected_labels = _ordered_region_labels(adjusted_input)
    if factor_set.region_labels != expected_labels:
        raise ValueError("SPH factor region labels do not match the diffusion input")
    if factor_set.group_count != adjusted_input.group_count:
        raise ValueError("SPH factor group count does not match the diffusion input")
    expected_fingerprint = sph_source_fingerprint(
        diffusion_input,
        spacing,
        factor_set.boundary_condition,
        reference_mode=factor_set.reference_mode,
        axial_sph_zones=factor_set.axial_sph_zones,
    )
    if factor_set.source_fingerprint != expected_fingerprint:
        raise ValueError(
            "SPH factors are stale for the selected MGXS, geometry, or mesh"
        )


def corrected_regions(
    diffusion_input: ConcentricDiffusionInput,
    factor_set: SphFactorSet,
    spacing: ConcentricMeshSpacing,
) -> dict[str, MultiGroupRegion]:
    validate_sph_factors(diffusion_input, spacing, factor_set)
    adjusted_input = sph_diffusion_input(diffusion_input, factor_set)
    factors_by_label = dict(
        zip(factor_set.region_labels, factor_set.factors, strict=True)
    )
    corrected: dict[str, MultiGroupRegion] = {}
    for label, region in adjusted_input.regions.items():
        factor = factors_by_label[label]
        corrected[label] = replace(
            region,
            diffusion=region.diffusion / factor,
            absorption=region.absorption * factor,
            nu_fission=region.nu_fission * factor,
            kappa_fission=region.kappa_fission * factor,
            scatter=region.scatter * factor[:, None],
            chi=region.chi.copy(),
        )
    return corrected


def build_sph_corrected_system(
    diffusion_input: ConcentricDiffusionInput,
    factor_set: SphFactorSet,
    spacing: ConcentricMeshSpacing,
) -> MultiGroupDiffusionSystem:
    adjusted_input = sph_diffusion_input(diffusion_input, factor_set)
    regions = corrected_regions(diffusion_input, factor_set, spacing)
    model = adjusted_input.build_model(
        regions=regions,
        boundary_condition=factor_set.boundary_condition,
    )
    mesh = build_concentric_mesh(model, spacing)
    return build_multigroup_2d_system(model, mesh=mesh, x_insert=0.0)


def region_integrated_flux(
    system: MultiGroupDiffusionSystem,
    solution: dict[str, Any],
    *,
    region_labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    labels = system.region_labels if region_labels is None else region_labels
    system_lookup = {
        label: index for index, label in enumerate(system.region_labels)
    }
    unknown = set(labels) - set(system_lookup)
    if unknown:
        raise ValueError(f"Unknown diffusion regions: {sorted(unknown)}")

    phi_groups = np.asarray(solution["phi_groups"], dtype=float)
    expected = (
        system.group_count,
        system.mesh.nr,
        system.mesh.nz,
    )
    if phi_groups.shape != expected:
        raise ValueError(
            f"Solution flux must have shape {expected}, got {phi_groups.shape}"
        )
    phi = phi_groups.reshape(system.group_count, system.cell_count).T
    weighted = phi * system.mesh.volumes[:, None]
    integrated = np.zeros((len(labels), system.group_count), dtype=float)
    for row, label in enumerate(labels):
        mask = system.region_index == system_lookup[label]
        integrated[row] = np.sum(weighted[mask], axis=0)
    return integrated


def region_integrated_power(
    system: MultiGroupDiffusionSystem,
    solution: dict[str, Any],
    *,
    region_labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    labels = system.region_labels if region_labels is None else region_labels
    system_lookup = {
        label: index for index, label in enumerate(system.region_labels)
    }
    unknown = set(labels) - set(system_lookup)
    if unknown:
        raise ValueError(f"Unknown diffusion regions: {sorted(unknown)}")

    phi = _solution_phi0(solution)
    power_rate = _power_density(system, phi) * system.mesh.volumes
    integrated = np.zeros(len(labels), dtype=float)
    for row, label in enumerate(labels):
        mask = system.region_index == system_lookup[label]
        integrated[row] = float(np.sum(power_rate[mask]))
    return integrated


def _find_zone_label_at(
    diffusion_input: ConcentricDiffusionInput,
    r_mid: float,
    z_mid: float,
) -> str | None:
    label = None
    for zone in diffusion_input.zones:
        if (
            zone.r_min <= r_mid < zone.r_max
            and zone.z_min <= z_mid < zone.z_max
        ):
            label = zone.region.name
    return label


def _cut_values(
    start: float,
    stop: float,
    boundaries: set[float],
) -> list[float]:
    cuts = [start, stop]
    cuts.extend(value for value in boundaries if start < value < stop)
    return sorted(cuts)


def _reference_mesh_region_power(
    diffusion_input: ConcentricDiffusionInput,
    region_labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    reference = diffusion_input.ce_reference
    if reference is None or reference.power_mesh is None:
        raise ValueError(
            "Axial power-shape SPH requires the CE kappa-fission power mesh"
        )
    mesh = reference.power_mesh
    label_lookup = {label: index for index, label in enumerate(region_labels)}
    power = np.zeros(len(region_labels), dtype=float)
    variance = np.zeros(len(region_labels), dtype=float)
    radial_boundaries = {
        value
        for zone in diffusion_input.zones
        for value in (zone.r_min, zone.r_max)
    }
    axial_boundaries = {
        value
        for zone in diffusion_input.zones
        for value in (zone.z_min, zone.z_max)
    }

    for radial_index, (r_min, r_max) in enumerate(
        zip(mesh.r_edges_cm, mesh.r_edges_cm[1:])
    ):
        radial_denominator = r_max * r_max - r_min * r_min
        if radial_denominator <= 0.0:
            continue
        r_cuts = _cut_values(r_min, r_max, radial_boundaries)
        for axial_index, (z_min, z_max) in enumerate(
            zip(mesh.z_edges_cm, mesh.z_edges_cm[1:])
        ):
            axial_denominator = z_max - z_min
            if axial_denominator <= 0.0:
                continue
            z_cuts = _cut_values(z_min, z_max, axial_boundaries)
            mean = float(mesh.mean[radial_index, axial_index])
            std_dev = float(mesh.std_dev[radial_index, axial_index])
            if mean == 0.0 and std_dev == 0.0:
                continue
            for sub_r_min, sub_r_max in zip(r_cuts, r_cuts[1:]):
                radial_fraction = (
                    sub_r_max * sub_r_max - sub_r_min * sub_r_min
                ) / radial_denominator
                r_mid = 0.5 * (sub_r_min + sub_r_max)
                for sub_z_min, sub_z_max in zip(z_cuts, z_cuts[1:]):
                    label = _find_zone_label_at(
                        diffusion_input,
                        r_mid,
                        0.5 * (sub_z_min + sub_z_max),
                    )
                    if label not in label_lookup:
                        continue
                    fraction = (
                        radial_fraction
                        * (sub_z_max - sub_z_min)
                        / axial_denominator
                    )
                    row = label_lookup[label]
                    power[row] += mean * fraction
                    variance[row] += (std_dev * fraction) ** 2
    return power, np.sqrt(variance)


def _normalized_positive(values: np.ndarray, name: str) -> np.ndarray:
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must contain positive total power")
    return values / total


def _build_sph_reference_from_flux(
    diffusion_input: ConcentricDiffusionInput,
    settings: SphSettings = SphSettings(),
    *,
    labels: tuple[str, ...],
    flux_values: Mapping[str, Any],
    exclude_base_labels: bool,
) -> SphReferenceData:
    missing = [label for label in labels if label not in flux_values]
    if missing:
        raise ValueError(
            "Continuous-energy reference is missing region/group flux for "
            f"{missing}"
        )
    flux = np.vstack(
        [flux_values[label].mean for label in labels]
    )
    std_dev = np.vstack(
        [flux_values[label].std_dev for label in labels]
    )
    nu_fission = np.vstack(
        [diffusion_input.regions[label].nu_fission for label in labels]
    )
    production = float(np.sum(nu_fission * flux))
    if not math.isfinite(production) or production <= 0.0:
        raise ValueError("Continuous-energy reference has zero fission production")
    flux = flux / production
    std_dev = std_dev / production

    group_max = np.max(flux, axis=0)
    relative_std = np.divide(
        std_dev,
        flux,
        out=np.full_like(std_dev, np.inf),
        where=flux > 0.0,
    )
    active = (
        (flux > settings.active_flux_fraction * group_max[None, :])
        & (relative_std <= settings.maximum_relative_std_dev)
    )
    excluded = set(settings.excluded_region_labels)
    if excluded:
        for row, label in enumerate(labels):
            if label in excluded or (
                exclude_base_labels and _base_region_label(label) in excluded
            ):
                active[row, :] = False
    if not np.any(active):
        raise ValueError(
            "Continuous-energy reference has no statistically active "
            "region/group flux bins"
        )
    return SphReferenceData(
        region_labels=labels,
        flux=flux,
        std_dev=std_dev,
        relative_std_dev=relative_std,
        active=active,
        fission_production=production,
    )


def build_sph_reference(
    diffusion_input: ConcentricDiffusionInput,
    settings: SphSettings = SphSettings(),
    *,
    region_labels: tuple[str, ...] | None = None,
) -> SphReferenceData:
    labels = (
        _ordered_region_labels(diffusion_input)
        if region_labels is None
        else region_labels
    )
    reference = diffusion_input.ce_reference
    if reference is None:
        raise ValueError(
            "SPH fitting requires raw continuous-energy region/group flux "
            "in the MGXS export"
        )
    return _build_sph_reference_from_flux(
        diffusion_input,
        settings,
        labels=labels,
        flux_values=reference.region_flux,
        exclude_base_labels=False,
    )


def build_axial_flux_sph_reference(
    diffusion_input: ConcentricDiffusionInput,
    settings: SphSettings = SphSettings(),
    *,
    axial_sph_zones: Mapping[str, Any],
    region_labels: tuple[str, ...] | None = None,
) -> SphReferenceData:
    adjusted_input = axialized_diffusion_input(diffusion_input, axial_sph_zones)
    labels = (
        _ordered_region_labels(adjusted_input)
        if region_labels is None
        else region_labels
    )
    reference = diffusion_input.ce_reference
    if reference is None:
        raise ValueError(
            "Axial flux SPH requires continuous-energy region/group flux"
        )
    flux_values = dict(reference.region_flux)
    flux_values.update(reference.axial_region_flux)
    missing_axial = [
        label
        for label in labels
        if AXIAL_SPH_LABEL_SEPARATOR in label
        and label not in reference.axial_region_flux
    ]
    if missing_axial:
        raise ValueError(
            "Continuous-energy reference is missing axial region/group flux "
            f"for {missing_axial}"
        )
    return _build_sph_reference_from_flux(
        adjusted_input,
        settings,
        labels=labels,
        flux_values=flux_values,
        exclude_base_labels=True,
    )


def build_axial_power_sph_reference(
    diffusion_input: ConcentricDiffusionInput,
    settings: SphSettings = SphSettings(),
    *,
    axial_sph_zones: Mapping[str, Any],
    region_labels: tuple[str, ...] | None = None,
) -> SphPowerReferenceData:
    adjusted_input = axialized_diffusion_input(diffusion_input, axial_sph_zones)
    labels = (
        _ordered_region_labels(adjusted_input)
        if region_labels is None
        else region_labels
    )
    power_raw, std_dev_raw = _reference_mesh_region_power(
        adjusted_input,
        labels,
    )
    total_power = float(np.sum(power_raw))
    power = _normalized_positive(power_raw, "CE power reference")
    std_dev = std_dev_raw / total_power
    relative_std = np.divide(
        std_dev,
        power,
        out=np.full_like(std_dev, np.inf),
        where=power > 0.0,
    )
    power_max = float(np.max(power))
    fissile = np.array(
        [
            bool(
                np.any(adjusted_input.regions[label].nu_fission > 0.0)
                or np.any(adjusted_input.regions[label].kappa_fission > 0.0)
            )
            for label in labels
        ],
        dtype=bool,
    )
    active = (
        fissile
        & (power > settings.active_flux_fraction * power_max)
        & (relative_std <= settings.maximum_relative_std_dev)
    )
    excluded = set(settings.excluded_region_labels)
    if excluded:
        for row, label in enumerate(labels):
            if label in excluded or _base_region_label(label) in excluded:
                active[row] = False
    if not np.any(active):
        raise ValueError(
            "Continuous-energy power reference has no active axial SPH zones"
        )
    return SphPowerReferenceData(
        region_labels=labels,
        power=power,
        std_dev=std_dev,
        relative_std_dev=relative_std,
        active=active,
        total_power=total_power,
    )


def _solution_phi0(solution: dict[str, Any]) -> np.ndarray:
    phi_groups = np.asarray(solution["phi_groups"], dtype=float)
    return phi_groups.reshape(phi_groups.shape[0], -1).T


def _solve_settings(settings: SphSettings) -> dict[str, Any]:
    return {
        "max_iter": settings.solver_max_iterations,
        "tol": settings.solver_tolerance,
        "source_tol": settings.solver_source_tolerance,
        "max_inner_iter": settings.solver_max_inner_iterations,
        "inner_tol": settings.solver_inner_tolerance,
    }


def _fit_region_flux_sph_factors(
    diffusion_input: ConcentricDiffusionInput,
    *,
    spacing: ConcentricMeshSpacing,
    settings: SphSettings = SphSettings(),
) -> SphFitResult:
    region_labels = _ordered_region_labels(diffusion_input)
    reference = build_sph_reference(
        diffusion_input,
        settings,
        region_labels=region_labels,
    )
    reference_flux = reference.flux
    active = reference.active
    factors = np.ones_like(reference_flux)
    fingerprint = sph_source_fingerprint(
        diffusion_input,
        spacing,
        settings.boundary_condition,
        reference_mode=REFERENCE_MODE_REGION_FLUX,
    )
    history: list[dict[str, float | int]] = []
    previous_k: float | None = None
    stable_count = 0
    final_system = None
    final_solution = None
    factor_history: list[np.ndarray] = []

    for iteration in range(1, settings.max_iterations + 1):
        factor_history.append(factors.copy())
        candidate = SphFactorSet(
            region_labels=region_labels,
            factors=factors,
            active=active,
            converged=False,
            iterations=iteration - 1,
            history=tuple(history),
            source_fingerprint=fingerprint,
            mesh_spacing=spacing.as_dict(),
            provisional=True,
            reference_mode=REFERENCE_MODE_REGION_FLUX,
            boundary_condition=settings.boundary_condition,
        )
        system = build_sph_corrected_system(
            diffusion_input,
            candidate,
            spacing,
        )
        solution = solve_multigroup_system(
            system,
            **_solve_settings(settings),
        )
        current_flux = region_integrated_flux(
            system,
            solution,
            region_labels=region_labels,
        )
        if np.any(active & (current_flux <= 0.0)):
            raise RuntimeError(
                "SPH iteration produced zero flux in an active region/group"
            )

        target = factors.copy()
        target[active] = reference_flux[active] / current_flux[active]
        corrected_flux = factors * current_flux
        relative_error = np.zeros_like(reference_flux)
        relative_error[active] = np.abs(
            corrected_flux[active] / reference_flux[active] - 1.0
        )
        max_flux_error = float(np.max(relative_error[active]))
        factor_log_change = np.zeros_like(reference_flux)
        factor_log_change[active] = np.abs(
            np.log(target[active]) - np.log(factors[active])
        )
        max_factor_log_change = float(np.max(factor_log_change[active]))
        k_eff = float(solution["k_eff"])
        k_change_pcm = (
            math.inf
            if previous_k is None
            else abs(k_eff - previous_k) * 1.0e5
        )
        history.append(
            {
                "iteration": iteration,
                "k_eff": k_eff,
                "k_change_pcm": k_change_pcm,
                "max_flux_error": max_flux_error,
                "max_factor_log_change": max_factor_log_change,
                "minimum_factor": float(np.min(factors)),
                "maximum_factor": float(np.max(factors)),
            }
        )
        final_system = system
        final_solution = solution

        if (
            max_flux_error <= settings.flux_tolerance
            and max_factor_log_change <= settings.factor_stability_tolerance
        ):
            stable_count += 1
            if stable_count >= settings.stable_iterations:
                break
        else:
            stable_count = 0

        log_factors = np.log(factors)
        log_target = np.log(target)
        log_factors[active] = (
            (1.0 - settings.damping) * log_factors[active]
            + settings.damping * log_target[active]
        )
        factors = np.clip(
            np.exp(log_factors),
            settings.minimum_factor,
            settings.maximum_factor,
        )
        previous_k = k_eff
    else:
        assert final_system is not None and final_solution is not None
        failed_factor_set = SphFactorSet(
            region_labels=region_labels,
            factors=factor_history[-1],
            active=active,
            converged=False,
            iterations=len(history),
            history=tuple(history),
            source_fingerprint=fingerprint,
            mesh_spacing=spacing.as_dict(),
            provisional=True,
            reference_mode=REFERENCE_MODE_REGION_FLUX,
            boundary_condition=settings.boundary_condition,
        )
        failed_result = SphFitResult(
            factors=failed_factor_set,
            system=final_system,
            solution=final_solution,
            qualification=evaluate_sph_qualification(
                diffusion_input,
                final_system,
                final_solution,
                failed_factor_set,
                settings=settings,
            ),
            factor_history=tuple(
                _readonly_array(values)
                for values in factor_history
            ),
        )
        raise SphConvergenceError(
            "SPH iteration did not converge within "
            f"{settings.max_iterations} iterations; "
            f"last max flux error={history[-1]['max_flux_error']:.6g}",
            failed_result,
        )

    assert final_system is not None and final_solution is not None
    provisional = (
        diffusion_input.ce_reference is None
        or diffusion_input.ce_reference.power_mesh is None
    )
    factor_set = SphFactorSet(
        region_labels=region_labels,
        factors=factors,
        active=active,
        converged=True,
        iterations=len(history),
        history=tuple(history),
        source_fingerprint=fingerprint,
        mesh_spacing=spacing.as_dict(),
        provisional=provisional,
        reference_mode=REFERENCE_MODE_REGION_FLUX,
        boundary_condition=settings.boundary_condition,
    )
    qualification = evaluate_sph_qualification(
        diffusion_input,
        final_system,
        final_solution,
        factor_set,
        settings=settings,
    )
    return SphFitResult(
        factors=factor_set,
        system=final_system,
        solution=final_solution,
        qualification=qualification,
        factor_history=tuple(
            _readonly_array(values)
            for values in factor_history
        ),
    )


def _fit_axial_region_flux_sph_factors(
    diffusion_input: ConcentricDiffusionInput,
    *,
    spacing: ConcentricMeshSpacing,
    settings: SphSettings,
    axial_sph_zones: Mapping[str, Any],
) -> SphFitResult:
    normalized_zones = _normalize_axial_sph_zones(axial_sph_zones)
    adjusted_input = axialized_diffusion_input(diffusion_input, normalized_zones)
    region_labels = _ordered_region_labels(adjusted_input)
    reference = build_axial_flux_sph_reference(
        diffusion_input,
        settings,
        axial_sph_zones=normalized_zones,
        region_labels=region_labels,
    )
    reference_flux = reference.flux
    active = reference.active
    factors = np.ones_like(reference_flux)
    fingerprint = sph_source_fingerprint(
        diffusion_input,
        spacing,
        settings.boundary_condition,
        reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
        axial_sph_zones=normalized_zones,
    )
    history: list[dict[str, float | int]] = []
    previous_k: float | None = None
    stable_count = 0
    final_system = None
    final_solution = None
    factor_history: list[np.ndarray] = []

    for iteration in range(1, settings.max_iterations + 1):
        factor_history.append(factors.copy())
        candidate = SphFactorSet(
            region_labels=region_labels,
            factors=factors,
            active=active,
            converged=False,
            iterations=iteration - 1,
            history=tuple(history),
            source_fingerprint=fingerprint,
            mesh_spacing=spacing.as_dict(),
            provisional=True,
            reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
            axial_sph_zones=normalized_zones,
            boundary_condition=settings.boundary_condition,
        )
        system = build_sph_corrected_system(
            diffusion_input,
            candidate,
            spacing,
        )
        solution = solve_multigroup_system(
            system,
            **_solve_settings(settings),
        )
        current_flux = region_integrated_flux(
            system,
            solution,
            region_labels=region_labels,
        )
        if np.any(active & (current_flux <= 0.0)):
            raise RuntimeError(
                "SPH iteration produced zero flux in an active axial region/group"
            )

        target = factors.copy()
        target[active] = reference_flux[active] / current_flux[active]
        corrected_flux = factors * current_flux
        relative_error = np.zeros_like(reference_flux)
        relative_error[active] = np.abs(
            corrected_flux[active] / reference_flux[active] - 1.0
        )
        max_flux_error = float(np.max(relative_error[active]))
        factor_log_change = np.zeros_like(reference_flux)
        factor_log_change[active] = np.abs(
            np.log(target[active]) - np.log(factors[active])
        )
        max_factor_log_change = float(np.max(factor_log_change[active]))
        k_eff = float(solution["k_eff"])
        k_change_pcm = (
            math.inf
            if previous_k is None
            else abs(k_eff - previous_k) * 1.0e5
        )
        history.append(
            {
                "iteration": iteration,
                "k_eff": k_eff,
                "k_change_pcm": k_change_pcm,
                "max_flux_error": max_flux_error,
                "max_factor_log_change": max_factor_log_change,
                "minimum_factor": float(np.min(factors)),
                "maximum_factor": float(np.max(factors)),
            }
        )
        final_system = system
        final_solution = solution

        if (
            max_flux_error <= settings.flux_tolerance
            and max_factor_log_change <= settings.factor_stability_tolerance
        ):
            stable_count += 1
            if stable_count >= settings.stable_iterations:
                break
        else:
            stable_count = 0

        log_factors = np.log(factors)
        log_target = np.log(target)
        log_factors[active] = (
            (1.0 - settings.damping) * log_factors[active]
            + settings.damping * log_target[active]
        )
        factors = np.clip(
            np.exp(log_factors),
            settings.minimum_factor,
            settings.maximum_factor,
        )
        previous_k = k_eff
    else:
        assert final_system is not None and final_solution is not None
        failed_factor_set = SphFactorSet(
            region_labels=region_labels,
            factors=factor_history[-1],
            active=active,
            converged=False,
            iterations=len(history),
            history=tuple(history),
            source_fingerprint=fingerprint,
            mesh_spacing=spacing.as_dict(),
            provisional=True,
            reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
            axial_sph_zones=normalized_zones,
            boundary_condition=settings.boundary_condition,
        )
        failed_result = SphFitResult(
            factors=failed_factor_set,
            system=final_system,
            solution=final_solution,
            qualification=evaluate_sph_qualification(
                diffusion_input,
                final_system,
                final_solution,
                failed_factor_set,
                settings=settings,
            ),
            factor_history=tuple(
                _readonly_array(values)
                for values in factor_history
            ),
        )
        raise SphConvergenceError(
            "SPH iteration did not converge within "
            f"{settings.max_iterations} iterations; "
            f"last max flux error={history[-1]['max_flux_error']:.6g}",
            failed_result,
        )

    assert final_system is not None and final_solution is not None
    provisional = (
        diffusion_input.ce_reference is None
        or diffusion_input.ce_reference.power_mesh is None
    )
    factor_set = SphFactorSet(
        region_labels=region_labels,
        factors=factors,
        active=active,
        converged=True,
        iterations=len(history),
        history=tuple(history),
        source_fingerprint=fingerprint,
        mesh_spacing=spacing.as_dict(),
        provisional=provisional,
        reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
        axial_sph_zones=normalized_zones,
        boundary_condition=settings.boundary_condition,
    )
    qualification = evaluate_sph_qualification(
        diffusion_input,
        final_system,
        final_solution,
        factor_set,
        settings=settings,
    )
    return SphFitResult(
        factors=factor_set,
        system=final_system,
        solution=final_solution,
        qualification=qualification,
        factor_history=tuple(
            _readonly_array(values)
            for values in factor_history
        ),
    )


def _fit_axial_power_sph_factors(
    diffusion_input: ConcentricDiffusionInput,
    *,
    spacing: ConcentricMeshSpacing,
    settings: SphSettings,
    axial_sph_zones: Mapping[str, Any],
) -> SphFitResult:
    normalized_zones = _normalize_axial_sph_zones(axial_sph_zones)
    adjusted_input = axialized_diffusion_input(diffusion_input, normalized_zones)
    region_labels = _ordered_region_labels(adjusted_input)
    reference = build_axial_power_sph_reference(
        diffusion_input,
        settings,
        axial_sph_zones=normalized_zones,
        region_labels=region_labels,
    )
    active_rows = reference.active
    active = np.repeat(
        active_rows[:, None],
        diffusion_input.group_count,
        axis=1,
    )
    factors = np.ones((len(region_labels), diffusion_input.group_count))
    fingerprint = sph_source_fingerprint(
        diffusion_input,
        spacing,
        settings.boundary_condition,
        reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
        axial_sph_zones=normalized_zones,
    )
    history: list[dict[str, float | int]] = []
    previous_k: float | None = None
    stable_count = 0
    final_system = None
    final_solution = None
    factor_history: list[np.ndarray] = []

    for iteration in range(1, settings.max_iterations + 1):
        factor_history.append(factors.copy())
        candidate = SphFactorSet(
            region_labels=region_labels,
            factors=factors,
            active=active,
            converged=False,
            iterations=iteration - 1,
            history=tuple(history),
            source_fingerprint=fingerprint,
            mesh_spacing=spacing.as_dict(),
            provisional=True,
            reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
            axial_sph_zones=normalized_zones,
            boundary_condition=settings.boundary_condition,
        )
        system = build_sph_corrected_system(
            diffusion_input,
            candidate,
            spacing,
        )
        solution = solve_multigroup_system(
            system,
            **_solve_settings(settings),
        )
        current_power = _normalized_positive(
            region_integrated_power(
                system,
                solution,
                region_labels=region_labels,
            ),
            "Diffusion axial-zone power",
        )
        if np.any(current_power[active_rows] <= 0.0):
            raise RuntimeError(
                "SPH iteration produced zero power in an active axial zone"
            )

        correction = np.ones_like(current_power)
        correction[active_rows] = (
            reference.power[active_rows] / current_power[active_rows]
        )
        target = factors.copy()
        target[active_rows, :] = (
            factors[active_rows, :] * correction[active_rows, None]
        )
        relative_error = np.zeros_like(current_power)
        relative_error[active_rows] = np.abs(
            current_power[active_rows] / reference.power[active_rows] - 1.0
        )
        max_power_error = float(np.max(relative_error[active_rows]))
        factor_log_change = np.zeros_like(factors)
        factor_log_change[active] = np.abs(
            np.log(target[active]) - np.log(factors[active])
        )
        max_factor_log_change = float(np.max(factor_log_change[active]))
        k_eff = float(solution["k_eff"])
        k_change_pcm = (
            math.inf
            if previous_k is None
            else abs(k_eff - previous_k) * 1.0e5
        )
        history.append(
            {
                "iteration": iteration,
                "k_eff": k_eff,
                "k_change_pcm": k_change_pcm,
                "max_flux_error": max_power_error,
                "max_power_error": max_power_error,
                "max_factor_log_change": max_factor_log_change,
                "minimum_factor": float(np.min(factors)),
                "maximum_factor": float(np.max(factors)),
            }
        )
        final_system = system
        final_solution = solution

        if (
            max_power_error <= settings.flux_tolerance
            and max_factor_log_change <= settings.factor_stability_tolerance
        ):
            stable_count += 1
            if stable_count >= settings.stable_iterations:
                break
        else:
            stable_count = 0

        log_factors = np.log(factors)
        log_target = np.log(target)
        log_factors[active] = (
            (1.0 - settings.damping) * log_factors[active]
            + settings.damping * log_target[active]
        )
        factors = np.clip(
            np.exp(log_factors),
            settings.minimum_factor,
            settings.maximum_factor,
        )
        previous_k = k_eff
    else:
        assert final_system is not None and final_solution is not None
        failed_factor_set = SphFactorSet(
            region_labels=region_labels,
            factors=factor_history[-1],
            active=active,
            converged=False,
            iterations=len(history),
            history=tuple(history),
            source_fingerprint=fingerprint,
            mesh_spacing=spacing.as_dict(),
            provisional=True,
            reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
            axial_sph_zones=normalized_zones,
            boundary_condition=settings.boundary_condition,
        )
        failed_result = SphFitResult(
            factors=failed_factor_set,
            system=final_system,
            solution=final_solution,
            qualification=evaluate_sph_qualification(
                diffusion_input,
                final_system,
                final_solution,
                failed_factor_set,
                settings=settings,
            ),
            factor_history=tuple(
                _readonly_array(values)
                for values in factor_history
            ),
        )
        raise SphConvergenceError(
            "SPH iteration did not converge within "
            f"{settings.max_iterations} iterations; "
            f"last max power error={history[-1]['max_power_error']:.6g}",
            failed_result,
        )

    assert final_system is not None and final_solution is not None
    factor_set = SphFactorSet(
        region_labels=region_labels,
        factors=factors,
        active=active,
        converged=True,
        iterations=len(history),
        history=tuple(history),
        source_fingerprint=fingerprint,
        mesh_spacing=spacing.as_dict(),
        provisional=False,
        reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
        axial_sph_zones=normalized_zones,
        boundary_condition=settings.boundary_condition,
    )
    qualification = evaluate_sph_qualification(
        diffusion_input,
        final_system,
        final_solution,
        factor_set,
        settings=settings,
    )
    return SphFitResult(
        factors=factor_set,
        system=final_system,
        solution=final_solution,
        qualification=qualification,
        factor_history=tuple(
            _readonly_array(values)
            for values in factor_history
        ),
    )


def fit_sph_factors(
    diffusion_input: ConcentricDiffusionInput,
    *,
    spacing: ConcentricMeshSpacing,
    settings: SphSettings = SphSettings(),
    reference_mode: str = REFERENCE_MODE_REGION_FLUX,
    axial_sph_zones: Mapping[str, Any] | None = None,
) -> SphFitResult:
    mode = _normalize_reference_mode(reference_mode)
    if mode == REFERENCE_MODE_REGION_FLUX:
        if _normalize_axial_sph_zones(axial_sph_zones):
            raise ValueError("Region-flux SPH does not accept axial_sph_zones")
        return _fit_region_flux_sph_factors(
            diffusion_input,
            spacing=spacing,
            settings=settings,
        )
    if mode == REFERENCE_MODE_AXIAL_REGION_FLUX:
        return _fit_axial_region_flux_sph_factors(
            diffusion_input,
            spacing=spacing,
            settings=settings,
            axial_sph_zones=axial_sph_zones or {},
        )
    return _fit_axial_power_sph_factors(
        diffusion_input,
        spacing=spacing,
        settings=settings,
        axial_sph_zones=axial_sph_zones or {},
    )


def normalized_power_profiles(
    system: MultiGroupDiffusionSystem,
    solution: dict[str, Any],
) -> dict[str, np.ndarray]:
    phi = _solution_phi0(solution)
    power = (
        _power_density(system, phi)
        * system.mesh.volumes
    ).reshape(system.mesh.nr, system.mesh.nz)
    radial = np.sum(power, axis=1) / np.diff(system.mesh.r_edges)
    axial = np.sum(power, axis=0) / np.diff(system.mesh.z_edges)
    radial_scale = float(np.max(radial))
    axial_scale = float(np.max(axial))
    if radial_scale <= 0.0 or axial_scale <= 0.0:
        raise ValueError("Power profiles require positive fission power")
    return {
        "radial_axis_cm": system.mesh.r_grid,
        "radial": radial / radial_scale,
        "axial_axis_cm": system.mesh.z_grid,
        "axial": axial / axial_scale,
    }


def _profile_errors(
    reference_axis: np.ndarray,
    reference_values: np.ndarray,
    candidate_axis: np.ndarray,
    candidate_values: np.ndarray,
) -> dict[str, float]:
    interpolated = np.interp(
        reference_axis,
        candidate_axis,
        candidate_values,
        left=candidate_values[0],
        right=candidate_values[-1],
    )
    difference = interpolated - reference_values
    return {
        "rms": float(np.sqrt(np.mean(difference**2))),
        "maximum": float(np.max(np.abs(difference))),
    }


def qualify_mesh(
    diffusion_input: ConcentricDiffusionInput,
    *,
    reference_spacing: ConcentricMeshSpacing,
    candidate_spacing: ConcentricMeshSpacing,
    k_tolerance_pcm: float = 20.0,
    flux_tolerance: float = 5.0e-3,
    power_rms_tolerance: float = 1.0e-2,
) -> dict[str, Any]:
    reference_system = build_multigroup_2d_system(
        diffusion_input.build_model(),
        spacing=reference_spacing,
    )
    reference_solution = solve_multigroup_system(reference_system)
    candidate_system = build_multigroup_2d_system(
        diffusion_input.build_model(),
        spacing=candidate_spacing,
    )
    candidate_solution = solve_multigroup_system(candidate_system)
    labels = _ordered_region_labels(diffusion_input)
    reference_flux = region_integrated_flux(
        reference_system,
        reference_solution,
        region_labels=labels,
    )
    candidate_flux = region_integrated_flux(
        candidate_system,
        candidate_solution,
        region_labels=labels,
    )
    group_max = np.max(reference_flux, axis=0)
    active = reference_flux > 1.0e-10 * group_max[None, :]
    relative_flux_error = np.zeros_like(reference_flux)
    relative_flux_error[active] = np.abs(
        candidate_flux[active] / reference_flux[active] - 1.0
    )
    maximum_flux_error = float(np.max(relative_flux_error[active]))
    maximum_flux_index = np.unravel_index(
        np.argmax(relative_flux_error),
        relative_flux_error.shape,
    )

    reference_power = normalized_power_profiles(
        reference_system,
        reference_solution,
    )
    candidate_power = normalized_power_profiles(
        candidate_system,
        candidate_solution,
    )
    radial_error = _profile_errors(
        reference_power["radial_axis_cm"],
        reference_power["radial"],
        candidate_power["radial_axis_cm"],
        candidate_power["radial"],
    )
    axial_error = _profile_errors(
        reference_power["axial_axis_cm"],
        reference_power["axial"],
        candidate_power["axial_axis_cm"],
        candidate_power["axial"],
    )
    k_difference_pcm = (
        float(candidate_solution["k_eff"])
        - float(reference_solution["k_eff"])
    ) * 1.0e5
    accepted = (
        abs(k_difference_pcm) <= k_tolerance_pcm
        and maximum_flux_error <= flux_tolerance
        and radial_error["rms"] <= power_rms_tolerance
        and axial_error["rms"] <= power_rms_tolerance
    )
    return {
        "accepted": accepted,
        "reference_spacing": reference_spacing.as_dict(),
        "candidate_spacing": candidate_spacing.as_dict(),
        "reference": {
            "cell_count": reference_system.cell_count,
            "k_eff": float(reference_solution["k_eff"]),
            "timings_s": reference_solution["timings_s"],
        },
        "candidate": {
            "cell_count": candidate_system.cell_count,
            "k_eff": float(candidate_solution["k_eff"]),
            "timings_s": candidate_solution["timings_s"],
        },
        "k_difference_pcm": k_difference_pcm,
        "maximum_region_group_flux_error": maximum_flux_error,
        "maximum_region_group_flux_error_location": {
            "region": labels[maximum_flux_index[0]],
            "group": int(maximum_flux_index[1] + 1),
        },
        "radial_power_error": radial_error,
        "axial_power_error": axial_error,
        "thresholds": {
            "k_pcm": k_tolerance_pcm,
            "region_group_flux": flux_tolerance,
            "power_rms": power_rms_tolerance,
        },
    }


def _reference_power_profiles(
    diffusion_input: ConcentricDiffusionInput,
) -> dict[str, np.ndarray] | None:
    reference = diffusion_input.ce_reference
    if reference is None or reference.power_mesh is None:
        return None
    mesh = reference.power_mesh
    radial = np.sum(mesh.mean, axis=1) / np.diff(mesh.r_edges_cm)
    axial = np.sum(mesh.mean, axis=0) / np.diff(mesh.z_edges_cm)
    if np.max(radial) <= 0.0 or np.max(axial) <= 0.0:
        raise ValueError("Continuous-energy power mesh contains no fission power")
    return {
        "radial_axis_cm": 0.5 * (
            mesh.r_edges_cm[:-1] + mesh.r_edges_cm[1:]
        ),
        "radial": radial / np.max(radial),
        "axial_axis_cm": 0.5 * (
            mesh.z_edges_cm[:-1] + mesh.z_edges_cm[1:]
        ),
        "axial": axial / np.max(axial),
    }


def _evaluate_region_flux_sph_qualification(
    diffusion_input: ConcentricDiffusionInput,
    system: MultiGroupDiffusionSystem,
    solution: dict[str, Any],
    factor_set: SphFactorSet,
    settings: SphSettings = SphSettings(),
) -> dict[str, Any]:
    validate_sph_factors(
        diffusion_input,
        ConcentricMeshSpacing(**factor_set.mesh_spacing),
        factor_set,
    )
    if factor_set.reference_mode == REFERENCE_MODE_AXIAL_REGION_FLUX:
        reference = build_axial_flux_sph_reference(
            diffusion_input,
            settings,
            axial_sph_zones=factor_set.axial_sph_zones,
            region_labels=factor_set.region_labels,
        )
    else:
        reference = build_sph_reference(
            diffusion_input,
            settings,
            region_labels=factor_set.region_labels,
        )
    reference_flux = reference.flux
    active = factor_set.active
    if active.shape != reference_flux.shape or not np.any(active):
        raise ValueError(
            "SPH factor active mask does not match the CE reference flux"
        )
    diffusion_flux = region_integrated_flux(
        system,
        solution,
        region_labels=factor_set.region_labels,
    )
    corrected_flux = factor_set.factors * diffusion_flux
    errors = np.zeros_like(reference_flux)
    errors[active] = np.abs(
        corrected_flux[active] / reference_flux[active] - 1.0
    )
    maximum_flux_error = float(np.max(errors[active]))
    k_error_pcm = (
        float(solution["k_eff"])
        - diffusion_input.openmc_reference["keff"]
    ) * 1.0e5
    reference_std_pcm = (
        diffusion_input.openmc_reference["keff_std_dev"] * 1.0e5
    )

    power_reference = _reference_power_profiles(diffusion_input)
    radial_error = None
    axial_error = None
    if power_reference is not None:
        diffusion_power = normalized_power_profiles(system, solution)
        radial_error = _profile_errors(
            power_reference["radial_axis_cm"],
            power_reference["radial"],
            diffusion_power["radial_axis_cm"],
            diffusion_power["radial"],
        )
        axial_error = _profile_errors(
            power_reference["axial_axis_cm"],
            power_reference["axial"],
            diffusion_power["axial_axis_cm"],
            diffusion_power["axial"],
        )

    region_group_flux_accepted = (
        maximum_flux_error <= settings.qualification_flux_tolerance
    )
    radial_power_rms_accepted = (
        radial_error is not None
        and radial_error["rms"] <= settings.radial_power_rms_tolerance
    )
    axial_power_rms_accepted = (
        axial_error is not None
        and axial_error["rms"] <= settings.axial_power_rms_tolerance
    )
    qualified = (
        factor_set.converged
        and region_group_flux_accepted
        and radial_power_rms_accepted
        and axial_power_rms_accepted
    )
    return {
        "qualified": qualified,
        "provisional": factor_set.provisional,
        "reference_mode": factor_set.reference_mode,
        "factor_converged": factor_set.converged,
        "k_error_pcm": k_error_pcm,
        "reference_keff_std_dev_pcm": reference_std_pcm,
        "maximum_region_group_flux_error": maximum_flux_error,
        "radial_power_error": radial_error,
        "axial_power_error": axial_error,
        "solve_time_s": float(
            solution.get("timings_s", {}).get("total", math.nan)
        ),
        "acceptance_criteria": {
            "region_group_flux": region_group_flux_accepted,
            "radial_power_rms": radial_power_rms_accepted,
            "axial_power_rms": axial_power_rms_accepted,
        },
        "thresholds": {
            "region_group_flux": settings.qualification_flux_tolerance,
            "radial_power_rms": settings.radial_power_rms_tolerance,
            "axial_power_rms": settings.axial_power_rms_tolerance,
        },
    }


def _evaluate_axial_power_sph_qualification(
    diffusion_input: ConcentricDiffusionInput,
    system: MultiGroupDiffusionSystem,
    solution: dict[str, Any],
    factor_set: SphFactorSet,
    settings: SphSettings,
) -> dict[str, Any]:
    validate_sph_factors(
        diffusion_input,
        ConcentricMeshSpacing(**factor_set.mesh_spacing),
        factor_set,
    )
    reference = build_axial_power_sph_reference(
        diffusion_input,
        settings,
        axial_sph_zones=factor_set.axial_sph_zones,
        region_labels=factor_set.region_labels,
    )
    active_rows = np.any(factor_set.active, axis=1)
    if active_rows.shape != reference.power.shape or not np.any(active_rows):
        raise ValueError(
            "SPH factor active mask does not match the axial power reference"
        )
    diffusion_power = _normalized_positive(
        region_integrated_power(
            system,
            solution,
            region_labels=factor_set.region_labels,
        ),
        "Diffusion axial-zone power",
    )
    errors = np.zeros_like(reference.power)
    errors[active_rows] = np.abs(
        diffusion_power[active_rows] / reference.power[active_rows] - 1.0
    )
    maximum_power_error = float(np.max(errors[active_rows]))
    k_error_pcm = (
        float(solution["k_eff"])
        - diffusion_input.openmc_reference["keff"]
    ) * 1.0e5
    reference_std_pcm = (
        diffusion_input.openmc_reference["keff_std_dev"] * 1.0e5
    )

    power_reference = _reference_power_profiles(diffusion_input)
    radial_error = None
    axial_error = None
    if power_reference is not None:
        profile = normalized_power_profiles(system, solution)
        radial_error = _profile_errors(
            power_reference["radial_axis_cm"],
            power_reference["radial"],
            profile["radial_axis_cm"],
            profile["radial"],
        )
        axial_error = _profile_errors(
            power_reference["axial_axis_cm"],
            power_reference["axial"],
            profile["axial_axis_cm"],
            profile["axial"],
        )

    axial_zone_power_accepted = (
        maximum_power_error <= settings.qualification_flux_tolerance
    )
    radial_power_rms_accepted = (
        radial_error is not None
        and radial_error["rms"] <= settings.radial_power_rms_tolerance
    )
    axial_power_rms_accepted = (
        axial_error is not None
        and axial_error["rms"] <= settings.axial_power_rms_tolerance
    )
    qualified = (
        factor_set.converged
        and axial_zone_power_accepted
        and radial_power_rms_accepted
        and axial_power_rms_accepted
    )
    return {
        "qualified": qualified,
        "provisional": factor_set.provisional,
        "reference_mode": factor_set.reference_mode,
        "factor_converged": factor_set.converged,
        "k_error_pcm": k_error_pcm,
        "reference_keff_std_dev_pcm": reference_std_pcm,
        "maximum_axial_zone_power_error": maximum_power_error,
        "maximum_region_group_flux_error": maximum_power_error,
        "active_axial_power_zone_count": int(np.count_nonzero(active_rows)),
        "radial_power_error": radial_error,
        "axial_power_error": axial_error,
        "solve_time_s": float(
            solution.get("timings_s", {}).get("total", math.nan)
        ),
        "acceptance_criteria": {
            "axial_zone_power": axial_zone_power_accepted,
            "radial_power_rms": radial_power_rms_accepted,
            "axial_power_rms": axial_power_rms_accepted,
        },
        "thresholds": {
            "axial_zone_power": settings.qualification_flux_tolerance,
            "radial_power_rms": settings.radial_power_rms_tolerance,
            "axial_power_rms": settings.axial_power_rms_tolerance,
        },
    }


def evaluate_sph_qualification(
    diffusion_input: ConcentricDiffusionInput,
    system: MultiGroupDiffusionSystem,
    solution: dict[str, Any],
    factor_set: SphFactorSet,
    settings: SphSettings = SphSettings(),
) -> dict[str, Any]:
    if factor_set.reference_mode == REFERENCE_MODE_REGION_FLUX:
        return _evaluate_region_flux_sph_qualification(
            diffusion_input,
            system,
            solution,
            factor_set,
            settings,
        )
    if factor_set.reference_mode == REFERENCE_MODE_AXIAL_REGION_FLUX:
        return _evaluate_region_flux_sph_qualification(
            diffusion_input,
            system,
            solution,
            factor_set,
            settings,
        )
    if factor_set.reference_mode == REFERENCE_MODE_AXIAL_POWER_SHAPE:
        return _evaluate_axial_power_sph_qualification(
            diffusion_input,
            system,
            solution,
            factor_set,
            settings,
        )
    raise ValueError(f"Unsupported SPH reference_mode {factor_set.reference_mode!r}")
