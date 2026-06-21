"""Lazy clean-core service for the four-group diffusion page."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .diffusion import rho_pcm
from .multigroup_diffusion import (
    BOUNDARY_GROUPWISE_FVM,
    ConcentricMeshSpacing,
    _power_density,
    build_multigroup_2d_system,
    solve_multigroup_system,
)
from .multigroup_diffusion_cache import (
    DEFAULT_CACHE_ROOT,
    DiffusionCacheSettings,
    PreparedDiffusionCase,
    prepare_concentric_diffusion_cache,
)
from .openmc_mgxs_adapter import load_concentric_diffusion_input
from .power_shape import (
    PowerShapeCorrection,
    apply_ce_power_shape_correction,
    apply_fixed_power_shape_factor,
)
from .reactivity import compute_reactivity
from .schemas import (
    MultigroupDiffusionGeometry,
    MultigroupDiffusionMetadata,
    MultigroupDiffusionProfile,
    MultigroupDiffusionResponse,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EXPORT_DIR = (
    _REPOSITORY_ROOT
    / "openmc"
    / "reference_data"
    / "concentric"
    / "group_sweep"
    / "group_4"
)
_DISPLAY_NR = 80
_DISPLAY_NZ = 100
_ROD_DELTA_ABSORPTION_CM_INV = 0.017
_ROD_CACHE_INSERTION_DECIMALS = 3
_NOTEBOOK_MESH_SPACING = ConcentricMeshSpacing(
    fuel_radial_cm=0.1,
    core_coolant_radial_cm=1.0,
    moderator_radial_cm=5.0,
    reflector_radial_cm=20.0,
    axial_cm=10.0,
)


@dataclass(frozen=True)
class _SolvedState:
    system: Any
    solution: dict[str, Any]
    cached: bool
    rod_insertion_percent: float


def _downsample_indices(size: int, target: int) -> np.ndarray:
    return np.round(
        np.linspace(0, size - 1, min(size, target))
    ).astype(int)


def _normalized(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return np.zeros_like(values)
    return values / maximum


def _scale_by(values: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values)
    return values / scale


def _mirror_radial_to_xz(values_rz: np.ndarray) -> np.ndarray:
    values = np.asarray(values_rz, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D radial-axial field")
    return np.concatenate((values[::-1, :], values), axis=0).T


def _mirror_radial_profile(
    axis_r_cm: np.ndarray,
    values_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis_r_cm, dtype=float).reshape(-1)
    values = np.asarray(values_r, dtype=float).reshape(-1)
    if axis.size != values.size:
        raise ValueError("Radial profile axis and values must have equal length")
    return (
        np.concatenate((-axis[::-1], axis)),
        np.concatenate((values[::-1], values)),
    )


class MultigroupDiffusionService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._prepared: PreparedDiffusionCase | None = None
        self._rod_cache: dict[float, tuple[Any, dict[str, Any]]] = {}
        self._clean_power_shape: PowerShapeCorrection | None = None

    def _paths(self) -> Path:
        export_dir = Path(
            os.environ.get(
                "MULTIGROUP_MGXS_EXPORT_DIR",
                str(_DEFAULT_EXPORT_DIR),
            )
        ).expanduser()
        return export_dir.resolve()

    def _initialize(self) -> None:
        if self._prepared is not None:
            return
        export_dir = self._paths()
        diffusion_input = load_concentric_diffusion_input(export_dir)
        if diffusion_input.group_count != 4:
            raise ValueError(
                "The online multigroup page requires a four-group MGXS export"
            )
        spacing = _NOTEBOOK_MESH_SPACING
        boundary_condition = os.environ.get(
            "MULTIGROUP_BOUNDARY_CONDITION",
            BOUNDARY_GROUPWISE_FVM,
        )
        cache_root = Path(
            os.environ.get(
                "MULTIGROUP_DIFFUSION_CACHE_DIR",
                str(DEFAULT_CACHE_ROOT),
            )
        )
        prepared = prepare_concentric_diffusion_cache(
            diffusion_input,
            settings=DiffusionCacheSettings(
                spacing=spacing,
                boundary_condition=boundary_condition,
                max_iter=300,
                tol=1.0e-5,
                source_tol=1.0e-3,
                max_inner_iter=200,
                inner_tol=1.0e-4,
            ),
            cache_root=cache_root,
            sph_factors=None,
        )
        self._prepared = prepared
        self._rod_cache[0.0] = (prepared.system, prepared.clean_solution)
        self._clean_power_shape = self._compute_clean_power_shape()

    def get_state(
        self,
        rod_insertion_percent: float = 0.0,
    ) -> MultigroupDiffusionResponse:
        with self._lock:
            self._initialize()
            state = self._solve_for_rod(rod_insertion_percent)
            return self._response(state)

    def recompute(
        self,
        rod_insertion_percent: float = 0.0,
    ) -> MultigroupDiffusionResponse:
        with self._lock:
            self._initialize()
            state = self._solve_for_rod(rod_insertion_percent, force=True)
            return self._response(state)

    def _solve_for_rod(
        self,
        rod_insertion_percent: float,
        *,
        force: bool = False,
    ) -> _SolvedState:
        assert self._prepared is not None
        clamped_percent = min(max(float(rod_insertion_percent), 0.0), 100.0)
        insertion = round(
            clamped_percent / 100.0,
            _ROD_CACHE_INSERTION_DECIMALS,
        )
        if not force and insertion in self._rod_cache:
            system, solution = self._rod_cache[insertion]
            return _SolvedState(
                system=system,
                solution=solution,
                cached=True,
                rod_insertion_percent=100.0 * insertion,
            )

        prepared = self._prepared
        if insertion == 0.0:
            system = prepared.system
            phi0 = prepared.clean_solution["phi_groups"]
        else:
            model = prepared.diffusion_input.build_model(
                delta_absorption_rod=_ROD_DELTA_ABSORPTION_CM_INV,
                boundary_condition=prepared.system.model.boundary_condition,
            )
            system = build_multigroup_2d_system(
                model,
                mesh=prepared.system.mesh,
                x_insert=insertion,
            )
            phi0 = prepared.clean_solution["phi_groups"]

        solution = solve_multigroup_system(
            system,
            phi0=phi0,
            max_iter=300,
            tol=1.0e-5,
            source_tol=1.0e-3,
            max_inner_iter=200,
            inner_tol=1.0e-4,
        )
        self._rod_cache[insertion] = (system, solution)
        if insertion == 0.0:
            self._rod_cache = {0.0: (system, solution)}
            self._clean_power_shape = self._compute_clean_power_shape(
                solution=solution,
            )
        return _SolvedState(
            system=system,
            solution=solution,
            cached=False,
            rod_insertion_percent=100.0 * insertion,
        )

    def _power_density_for(
        self,
        system: Any,
        solution: dict[str, Any],
    ) -> np.ndarray:
        phi = np.asarray(solution["phi_groups"], dtype=float).reshape(
            system.group_count,
            system.cell_count,
        ).T
        return _power_density(system, phi).reshape(system.mesh.nr, system.mesh.nz)

    def _compute_clean_power_shape(
        self,
        *,
        solution: dict[str, Any] | None = None,
    ) -> PowerShapeCorrection:
        assert self._prepared is not None
        prepared = self._prepared
        clean_solution = prepared.clean_solution if solution is None else solution
        power_density = self._power_density_for(prepared.system, clean_solution)
        volumes = prepared.system.mesh.volumes.reshape(
            prepared.system.mesh.nr,
            prepared.system.mesh.nz,
        )
        return apply_ce_power_shape_correction(
            power_density=power_density,
            volumes=volumes,
            r_edges_cm=prepared.system.mesh.r_edges,
            z_edges_cm=prepared.system.mesh.z_edges,
            reference=prepared.diffusion_input.ce_reference.power_mesh
            if prepared.diffusion_input.ce_reference is not None
            else None,
        )

    def _qualification(self, state: _SolvedState) -> dict[str, Any]:
        assert self._prepared is not None
        if state.system.x_insert != 0.0:
            return {
                "qualified": False,
                "provisional": True,
                "reason": (
                    "Rodded multigroup maps use an equivalent absorber and a "
                    "fixed clean-core CE power-shape correction; they are "
                    "visualization outputs, not a qualified rodded MGXS result."
                ),
            }
        if self._prepared.sph_factors is None:
            reference = self._prepared.diffusion_input.openmc_reference
            difference_pcm = abs(
                (float(state.solution["k_eff"]) - reference["keff"]) * 1.0e5
            )
            tolerance_pcm = max(10.0, reference["keff_std_dev"] * 3.0e5)
            qualified = difference_pcm <= tolerance_pcm
            return {
                "qualified": qualified,
                "provisional": not qualified,
                "reason": (
                    "Clean four-group diffusion uses the notebook mesh and "
                    "groupwise finite-volume boundary; power is corrected "
                    "with the clean continuous-energy OpenMC shape."
                ),
                "difference_pcm": difference_pcm,
                "tolerance_pcm": tolerance_pcm,
            }
        raise RuntimeError("The Core service no longer applies SPH factors")

    def _response(self, state: _SolvedState) -> MultigroupDiffusionResponse:
        assert self._prepared is not None
        assert self._clean_power_shape is not None
        prepared = self._prepared
        system = state.system
        solution = state.solution
        phi_groups = np.asarray(solution["phi_groups"], dtype=float)
        total_flux = np.sum(phi_groups, axis=0)
        clean_phi_groups = np.asarray(
            prepared.clean_solution["phi_groups"],
            dtype=float,
        )
        clean_total_flux = np.sum(clean_phi_groups, axis=0)
        power_density = self._power_density_for(system, solution)
        volumes = system.mesh.volumes.reshape(
            system.mesh.nr,
            system.mesh.nz,
        )
        if system.x_insert == 0.0:
            power_shape = self._clean_power_shape
        else:
            power_shape = apply_fixed_power_shape_factor(
                power_density=power_density,
                volumes=volumes,
                correction_factor=self._clean_power_shape.correction_factor,
            )
        displayed_power_density = power_shape.corrected_power_density
        power_rate = power_shape.corrected_power_rate

        moderator_radius = float(
            prepared.diffusion_input.geometry["moderator_radius_cm"]
        )
        moderator_edges = np.flatnonzero(
            np.isclose(
                system.mesh.r_edges,
                moderator_radius,
                rtol=0.0,
                atol=1.0e-8,
            )
        )
        radial_display_count = (
            int(moderator_edges[-1])
            if moderator_edges.size
            else system.mesh.nr
        )
        r_indices = _downsample_indices(radial_display_count, _DISPLAY_NR)
        z_indices = _downsample_indices(system.mesh.nz, _DISPLAY_NZ)
        clean_flux_display = clean_total_flux[np.ix_(r_indices, z_indices)]
        clean_power_display = self._clean_power_shape.corrected_power_density[
            np.ix_(r_indices, z_indices)
        ]
        displayed_flux_rz = _scale_by(
            total_flux[np.ix_(r_indices, z_indices)],
            float(np.max(clean_flux_display)),
        )
        displayed_power_rz = _scale_by(
            displayed_power_density[np.ix_(r_indices, z_indices)],
            float(np.max(clean_power_display)),
        )
        displayed_flux = _mirror_radial_to_xz(displayed_flux_rz)
        displayed_power = _mirror_radial_to_xz(displayed_power_rz)
        heatmap_x_cm = np.concatenate(
            (
                -system.mesh.r_grid[r_indices][::-1],
                system.mesh.r_grid[r_indices],
            )
        )
        radial_widths = np.diff(system.mesh.r_edges)
        axial_widths = np.diff(system.mesh.z_edges)
        midplane_index = int(np.argmin(np.abs(system.mesh.z_grid)))
        probe_radius = 0.5 * float(prepared.diffusion_input.geometry["core_radius_cm"])
        probe_radial_index = int(np.argmin(np.abs(system.mesh.r_grid - probe_radius)))
        radial_profile = slice(0, radial_display_count)
        radial_flux_axis, radial_flux_values = _mirror_radial_profile(
            system.mesh.r_grid[radial_profile],
            _scale_by(
                total_flux[radial_profile, midplane_index],
                float(np.max(clean_total_flux[radial_profile, midplane_index])),
            ),
        )
        axial_flux = _scale_by(
            total_flux[probe_radial_index, :],
            float(np.max(clean_total_flux[probe_radial_index, :])),
        )
        radial_power_axis, radial_power_values = _mirror_radial_profile(
            system.mesh.r_grid[radial_profile],
            _normalized(
                np.sum(power_rate[radial_profile, :], axis=1)
                / radial_widths[radial_profile]
            ),
        )
        axial_power = _normalized(
            np.sum(power_rate, axis=0) / axial_widths
        )

        qualification = self._qualification(state)
        reference = prepared.diffusion_input.openmc_reference
        spacing = prepared.manifest["settings"]["spacing"]
        factor_set = prepared.sph_factors
        return MultigroupDiffusionResponse(
            heatmapXCm=heatmap_x_cm.tolist(),
            heatmapZCm=system.mesh.z_grid[z_indices].tolist(),
            heatmapFlux=displayed_flux.tolist(),
            heatmapPower=displayed_power.tolist(),
            radialFlux=MultigroupDiffusionProfile(
                axisCm=radial_flux_axis.tolist(),
                values=radial_flux_values.tolist(),
            ),
            axialFlux=MultigroupDiffusionProfile(
                axisCm=system.mesh.z_grid.tolist(),
                values=axial_flux.tolist(),
            ),
            radialPower=MultigroupDiffusionProfile(
                axisCm=radial_power_axis.tolist(),
                values=radial_power_values.tolist(),
            ),
            axialPower=MultigroupDiffusionProfile(
                axisCm=system.mesh.z_grid.tolist(),
                values=axial_power.tolist(),
            ),
            energyGroupEdgesEv=list(
                prepared.diffusion_input.energy_group_edges_ev
            ),
            geometry=MultigroupDiffusionGeometry(
                coreRadiusCm=prepared.diffusion_input.geometry["core_radius_cm"],
                moderatorRadiusCm=prepared.diffusion_input.geometry[
                    "moderator_radius_cm"
                ],
                reflectorRadiusCm=prepared.diffusion_input.geometry[
                    "reflector_radius_cm"
                ],
                coreHeightCm=prepared.diffusion_input.geometry["core_height_cm"],
                outerHeightCm=prepared.diffusion_input.geometry["outer_height_cm"],
                resolvedRegionCount=len(prepared.diffusion_input.regions),
            ),
            metadata=MultigroupDiffusionMetadata(
                cleanCore=system.x_insert == 0.0,
                groupCount=system.group_count,
                kEff=float(solution["k_eff"]),
                reactivityPcm=rho_pcm(float(solution["k_eff"])),
                openmcCeReactivityPcm=compute_reactivity(
                    state.rod_insertion_percent,
                    False,
                ).totalPcm,
                openmcReferenceKEff=reference["keff"],
                openmcReferenceStdDevPcm=reference["keff_std_dev"] * 1.0e5,
                differencePcm=(
                    float(solution["k_eff"]) - reference["keff"]
                )
                * 1.0e5,
                iterations=int(solution["iterations"]),
                cached=state.cached,
                cellCount=system.cell_count,
                meshSpacingCm=spacing,
                timingsSeconds={
                    key: float(value)
                    for key, value in solution["timings_s"].items()
                },
                sphApplied=factor_set is not None,
                sphIterations=(
                    None if factor_set is None else factor_set.iterations
                ),
                provisional=bool(qualification.get("provisional", True)),
                qualified=bool(qualification.get("qualified", False)),
                qualification=qualification,
                rodInsertionPercent=state.rod_insertion_percent,
                rodDeltaAbsorptionCmInv=_ROD_DELTA_ABSORPTION_CM_INV,
                cleanCorrectionApplied=(
                    prepared.diffusion_input.ce_reference is not None
                    and prepared.diffusion_input.ce_reference.power_mesh is not None
                ),
                roddedSolveCached=state.cached,
                powerShapeCorrectionApplied=prepared.diffusion_input.ce_reference
                is not None
                and prepared.diffusion_input.ce_reference.power_mesh is not None,
                powerShapeCorrectionActiveBins=int(
                    np.count_nonzero(power_shape.active_bins)
                ),
                powerShapeCorrectionReferenceTotal=(
                    power_shape.reference_total
                    if power_shape.reference_total > 0.0
                    else None
                ),
                powerShapeCorrectionDiffusionTotal=(
                    power_shape.diffusion_total
                    if power_shape.diffusion_total > 0.0
                    else None
                ),
            ),
        )


multigroup_diffusion_service = MultigroupDiffusionService()
