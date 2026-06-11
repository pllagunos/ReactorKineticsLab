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
    ConcentricMeshSpacing,
    MultiGroupDiffusionSystem,
    MultiGroupRegion,
    _power_density,
    build_concentric_mesh,
    build_multigroup_2d_system,
    solve_multigroup_system,
)
from .openmc_mgxs_adapter import ConcentricDiffusionInput


SPH_ALGORITHM_VERSION = 1


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
    k_stability_pcm: float = 5.0
    stable_iterations: int = 2
    minimum_factor: float = 0.1
    maximum_factor: float = 10.0
    active_flux_fraction: float = 1.0e-10
    maximum_relative_std_dev: float = 0.5
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
        if self.flux_tolerance <= 0.0 or self.k_stability_pcm <= 0.0:
            raise ValueError("SPH convergence tolerances must be positive")
        if not 0.0 < self.minimum_factor < self.maximum_factor:
            raise ValueError("SPH factor bounds are invalid")
        if self.active_flux_fraction < 0.0:
            raise ValueError("SPH active-flux threshold must be non-negative")
        if self.maximum_relative_std_dev <= 0.0:
            raise ValueError("SPH relative uncertainty limit must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


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
    algorithm_version: int = SPH_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        factors = _readonly_array(self.factors)
        active = _readonly_array(self.active, dtype=bool)
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
        )


@dataclass(frozen=True)
class SphFitResult:
    factors: SphFactorSet
    system: MultiGroupDiffusionSystem
    solution: dict[str, Any]
    qualification: dict[str, Any]


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
) -> str:
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
    expected_labels = _ordered_region_labels(diffusion_input)
    if factor_set.region_labels != expected_labels:
        raise ValueError("SPH factor region labels do not match the diffusion input")
    if factor_set.group_count != diffusion_input.group_count:
        raise ValueError("SPH factor group count does not match the diffusion input")
    expected_fingerprint = sph_source_fingerprint(diffusion_input, spacing)
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
    factors_by_label = dict(
        zip(factor_set.region_labels, factor_set.factors, strict=True)
    )
    corrected: dict[str, MultiGroupRegion] = {}
    for label, region in diffusion_input.regions.items():
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
    regions = corrected_regions(diffusion_input, factor_set, spacing)
    model = diffusion_input.build_model(regions=regions)
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


def _normalized_reference_flux(
    diffusion_input: ConcentricDiffusionInput,
    region_labels: tuple[str, ...],
    settings: SphSettings,
) -> tuple[np.ndarray, np.ndarray, float]:
    reference = diffusion_input.ce_reference
    if reference is None:
        raise ValueError(
            "SPH fitting requires raw continuous-energy region/group flux "
            "in the MGXS export"
        )
    flux = np.vstack(
        [reference.region_flux[label].mean for label in region_labels]
    )
    std_dev = np.vstack(
        [reference.region_flux[label].std_dev for label in region_labels]
    )
    nu_fission = np.vstack(
        [diffusion_input.regions[label].nu_fission for label in region_labels]
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
    if not np.any(active):
        raise ValueError(
            "Continuous-energy reference has no statistically active "
            "region/group flux bins"
        )
    return flux, active, production


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


def fit_sph_factors(
    diffusion_input: ConcentricDiffusionInput,
    *,
    spacing: ConcentricMeshSpacing,
    settings: SphSettings = SphSettings(),
) -> SphFitResult:
    region_labels = _ordered_region_labels(diffusion_input)
    reference_flux, active, _ = _normalized_reference_flux(
        diffusion_input,
        region_labels,
        settings,
    )
    factors = np.ones_like(reference_flux)
    fingerprint = sph_source_fingerprint(diffusion_input, spacing)
    history: list[dict[str, float | int]] = []
    previous_k: float | None = None
    stable_count = 0
    final_system = None
    final_solution = None

    for iteration in range(1, settings.max_iterations + 1):
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
                "minimum_factor": float(np.min(factors)),
                "maximum_factor": float(np.max(factors)),
            }
        )
        final_system = system
        final_solution = solution

        if (
            max_flux_error <= settings.flux_tolerance
            and k_change_pcm <= settings.k_stability_pcm
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
        raise RuntimeError(
            "SPH iteration did not converge within "
            f"{settings.max_iterations} iterations; "
            f"last max flux error={history[-1]['max_flux_error']:.6g}"
        )

    assert final_system is not None and final_solution is not None
    reference_std_pcm = (
        diffusion_input.openmc_reference["keff_std_dev"] * 1.0e5
    )
    provisional = (
        reference_std_pcm > 15.0
        or diffusion_input.ce_reference is None
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
    )
    qualification = evaluate_sph_qualification(
        diffusion_input,
        final_system,
        final_solution,
        factor_set,
    )
    return SphFitResult(
        factors=factor_set,
        system=final_system,
        solution=final_solution,
        qualification=qualification,
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


def evaluate_sph_qualification(
    diffusion_input: ConcentricDiffusionInput,
    system: MultiGroupDiffusionSystem,
    solution: dict[str, Any],
    factor_set: SphFactorSet,
) -> dict[str, Any]:
    validate_sph_factors(
        diffusion_input,
        ConcentricMeshSpacing(**factor_set.mesh_spacing),
        factor_set,
    )
    reference_flux, _, _ = _normalized_reference_flux(
        diffusion_input,
        factor_set.region_labels,
        SphSettings(),
    )
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

    qualified = (
        factor_set.converged
        and not factor_set.provisional
        and abs(k_error_pcm) <= 50.0
        and maximum_flux_error <= 1.0e-2
        and radial_error is not None
        and axial_error is not None
        and radial_error["rms"] <= 2.0e-2
        and axial_error["rms"] <= 2.0e-2
        and radial_error["maximum"] <= 5.0e-2
        and axial_error["maximum"] <= 5.0e-2
        and reference_std_pcm <= 15.0
    )
    return {
        "qualified": qualified,
        "provisional": factor_set.provisional,
        "k_error_pcm": k_error_pcm,
        "reference_keff_std_dev_pcm": reference_std_pcm,
        "maximum_region_group_flux_error": maximum_flux_error,
        "radial_power_error": radial_error,
        "axial_power_error": axial_error,
        "thresholds": {
            "k_pcm": 50.0,
            "reference_keff_std_dev_pcm": 15.0,
            "region_group_flux": 1.0e-2,
            "power_rms": 2.0e-2,
            "power_maximum": 5.0e-2,
        },
    }
