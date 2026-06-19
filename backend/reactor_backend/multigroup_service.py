"""Lazy clean-core service for the four-group diffusion page."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .multigroup_diffusion import (
    BOUNDARY_EXTRAPOLATED_MESH,
    ConcentricMeshSpacing,
    _power_density,
    solve_multigroup_system,
)
from .multigroup_diffusion_cache import (
    DEFAULT_CACHE_ROOT,
    DiffusionCacheSettings,
    PreparedDiffusionCase,
    prepare_concentric_diffusion_cache,
)
from .multigroup_sph import (
    SphFactorSet,
    evaluate_sph_qualification,
    load_sph_factors,
)
from .openmc_mgxs_adapter import load_concentric_diffusion_input
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


def _spacing_from_factors(
    factors: SphFactorSet | None,
) -> ConcentricMeshSpacing:
    if factors is None:
        return ConcentricMeshSpacing()
    return ConcentricMeshSpacing(**factors.mesh_spacing)


def _downsample_indices(size: int, target: int) -> np.ndarray:
    return np.round(
        np.linspace(0, size - 1, min(size, target))
    ).astype(int)


def _normalized(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return np.zeros_like(values)
    return values / maximum


class MultigroupDiffusionService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._prepared: PreparedDiffusionCase | None = None
        self._solution: dict[str, Any] | None = None
        self._fresh = False

    def _paths(self) -> tuple[Path, Path]:
        export_dir = Path(
            os.environ.get(
                "MULTIGROUP_MGXS_EXPORT_DIR",
                str(_DEFAULT_EXPORT_DIR),
            )
        ).expanduser()
        factor_path = Path(
            os.environ.get(
                "MULTIGROUP_SPH_FACTORS_PATH",
                str(export_dir / "outputs" / "sph_factors.json"),
            )
        ).expanduser()
        return export_dir.resolve(), factor_path.resolve()

    def _initialize(self) -> None:
        if self._prepared is not None:
            return
        export_dir, factor_path = self._paths()
        diffusion_input = load_concentric_diffusion_input(export_dir)
        if diffusion_input.group_count != 4:
            raise ValueError(
                "The online multigroup page requires a four-group MGXS export"
            )
        factors = (
            load_sph_factors(factor_path)
            if factor_path.is_file()
            else None
        )
        spacing = _spacing_from_factors(factors)
        boundary_condition = os.environ.get(
            "MULTIGROUP_BOUNDARY_CONDITION",
            (
                factors.boundary_condition
                if factors is not None
                else BOUNDARY_EXTRAPOLATED_MESH
            ),
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
            ),
            cache_root=cache_root,
            sph_factors=factors,
        )
        self._prepared = prepared
        self._solution = prepared.clean_solution
        self._fresh = False

    def get_state(self) -> MultigroupDiffusionResponse:
        with self._lock:
            self._initialize()
            return self._response()

    def recompute(self) -> MultigroupDiffusionResponse:
        with self._lock:
            self._initialize()
            assert self._prepared is not None
            self._solution = solve_multigroup_system(
                self._prepared.system,
                max_iter=300,
                tol=1.0e-6,
                source_tol=1.0e-3,
                max_inner_iter=200,
                inner_tol=1.0e-4,
            )
            self._fresh = True
            return self._response()

    def _qualification(self) -> dict[str, Any]:
        assert self._prepared is not None
        assert self._solution is not None
        if self._prepared.sph_factors is None:
            return {
                "qualified": False,
                "provisional": True,
                "reason": (
                    "No matching SPH factor artifact is installed; displaying "
                    "the uncorrected clean-core solution."
                ),
            }
        return evaluate_sph_qualification(
            self._prepared.diffusion_input,
            self._prepared.system,
            self._solution,
            self._prepared.sph_factors,
        )

    def _response(self) -> MultigroupDiffusionResponse:
        assert self._prepared is not None
        assert self._solution is not None
        prepared = self._prepared
        system = prepared.system
        solution = self._solution
        phi_groups = np.asarray(solution["phi_groups"], dtype=float)
        total_flux = np.sum(phi_groups, axis=0)
        phi = phi_groups.reshape(system.group_count, system.cell_count).T
        power_density = _power_density(system, phi).reshape(
            system.mesh.nr,
            system.mesh.nz,
        )
        volumes = system.mesh.volumes.reshape(
            system.mesh.nr,
            system.mesh.nz,
        )
        flux_rate = total_flux * volumes
        power_rate = power_density * volumes

        r_indices = _downsample_indices(system.mesh.nr, _DISPLAY_NR)
        z_indices = _downsample_indices(system.mesh.nz, _DISPLAY_NZ)
        displayed_flux = _normalized(
            total_flux[np.ix_(r_indices, z_indices)]
        )
        displayed_power = _normalized(
            power_density[np.ix_(r_indices, z_indices)]
        )
        radial_widths = np.diff(system.mesh.r_edges)
        axial_widths = np.diff(system.mesh.z_edges)
        radial_flux = _normalized(
            np.sum(flux_rate, axis=1) / radial_widths
        )
        axial_flux = _normalized(
            np.sum(flux_rate, axis=0) / axial_widths
        )
        radial_power = _normalized(
            np.sum(power_rate, axis=1) / radial_widths
        )
        axial_power = _normalized(
            np.sum(power_rate, axis=0) / axial_widths
        )

        qualification = self._qualification()
        reference = prepared.diffusion_input.openmc_reference
        spacing = prepared.manifest["settings"]["spacing"]
        factor_set = prepared.sph_factors
        return MultigroupDiffusionResponse(
            heatmapRCm=system.mesh.r_grid[r_indices].tolist(),
            heatmapZCm=system.mesh.z_grid[z_indices].tolist(),
            heatmapFlux=displayed_flux.tolist(),
            heatmapPower=displayed_power.tolist(),
            radialFlux=MultigroupDiffusionProfile(
                axisCm=system.mesh.r_grid.tolist(),
                values=radial_flux.tolist(),
            ),
            axialFlux=MultigroupDiffusionProfile(
                axisCm=system.mesh.z_grid.tolist(),
                values=axial_flux.tolist(),
            ),
            radialPower=MultigroupDiffusionProfile(
                axisCm=system.mesh.r_grid.tolist(),
                values=radial_power.tolist(),
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
                cleanCore=True,
                groupCount=system.group_count,
                kEff=float(solution["k_eff"]),
                openmcReferenceKEff=reference["keff"],
                openmcReferenceStdDevPcm=reference["keff_std_dev"] * 1.0e5,
                differencePcm=(
                    float(solution["k_eff"]) - reference["keff"]
                )
                * 1.0e5,
                iterations=int(solution["iterations"]),
                cached=not self._fresh,
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
            ),
        )


multigroup_diffusion_service = MultigroupDiffusionService()
