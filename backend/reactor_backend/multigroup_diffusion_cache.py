"""Persistent preparation cache for resolved OpenMC diffusion inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .multigroup_diffusion import (
    ConcentricMeshSpacing,
    CylindricalMesh2D,
    MultiGroupDiffusionSystem,
    build_multigroup_2d_system,
    solve_multigroup_system,
)
from .openmc_mgxs_adapter import ConcentricDiffusionInput
from .multigroup_sph import (
    SphFactorSet,
    build_sph_corrected_system,
    corrected_regions,
)


CACHE_SCHEMA_VERSION = 4
DEFAULT_CACHE_ROOT = (
    Path(__file__).resolve().parents[1] / ".cache" / "openmc_diffusion"
)


@dataclass(frozen=True)
class DiffusionCacheSettings:
    spacing: ConcentricMeshSpacing = ConcentricMeshSpacing()
    delta_absorption_rod: float = 0.25
    max_iter: int = 300
    tol: float = 1.0e-6
    source_tol: float = 1.0e-3
    max_inner_iter: int = 200
    inner_tol: float = 1.0e-4

    def as_dict(self) -> dict[str, Any]:
        return {
            "spacing": self.spacing.as_dict(),
            "delta_absorption_rod": float(self.delta_absorption_rod),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "source_tol": float(self.source_tol),
            "max_inner_iter": int(self.max_inner_iter),
            "inner_tol": float(self.inner_tol),
        }


@dataclass(frozen=True)
class PreparedDiffusionCase:
    diffusion_input: ConcentricDiffusionInput
    system: MultiGroupDiffusionSystem
    clean_solution: dict[str, Any]
    cache_dir: Path
    cache_hit: bool
    fingerprint: str
    manifest: dict[str, Any]
    sph_factors: SphFactorSet | None

    def summary(self) -> dict[str, Any]:
        return {
            "cache_dir": str(self.cache_dir),
            "cache_hit": self.cache_hit,
            "fingerprint": self.fingerprint,
            "group_count": self.system.group_count,
            "cell_count": self.system.cell_count,
            "mesh": {
                "Nr": self.system.mesh.nr,
                "Nz": self.system.mesh.nz,
            },
            "sph": (
                None
                if self.sph_factors is None
                else {
                    "applied": True,
                    "converged": self.sph_factors.converged,
                    "provisional": self.sph_factors.provisional,
                    "iterations": self.sph_factors.iterations,
                }
            ),
            "clean_solution": {
                "k_eff": self.clean_solution["k_eff"],
                "iterations": self.clean_solution["iterations"],
                "balance_residual": self.clean_solution["balance_residual"],
                "timings_s": self.clean_solution.get("timings_s", {}),
            },
        }


def _fingerprint(
    diffusion_input: ConcentricDiffusionInput,
    settings: DiffusionCacheSettings,
    sph_factors: SphFactorSet | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"schema:{CACHE_SCHEMA_VERSION}\n".encode())
    for path in (
        diffusion_input.mgxs_json_path,
        diffusion_input.model_xml_path,
    ):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(
        json.dumps(
            settings.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    if sph_factors is not None:
        digest.update(
            json.dumps(
                sph_factors.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def _solution_arrays(solution: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "phi": np.asarray(solution["phi"]),
        "phi_groups": np.asarray(solution["phi_groups"]),
        "power_density": np.asarray(solution["power_density"]),
        "r_grid": np.asarray(solution["r_grid"]),
        "z_grid": np.asarray(solution["z_grid"]),
        "r_edges": np.asarray(solution["r_edges"]),
        "z_edges": np.asarray(solution["z_edges"]),
        "inner_iterations": np.asarray(
            solution["inner_iterations"], dtype=np.int64
        ),
    }


def _solution_metadata(solution: dict[str, Any]) -> dict[str, Any]:
    return {
        "k_eff": float(solution["k_eff"]),
        "Nr": int(solution["Nr"]),
        "Nz": int(solution["Nz"]),
        "cell_count": int(solution["cell_count"]),
        "group_count": int(solution["group_count"]),
        "iterations": int(solution["iterations"]),
        "converged": bool(solution["converged"]),
        "balance_residual": float(solution["balance_residual"]),
        "timings_s": {
            name: float(value)
            for name, value in solution.get("timings_s", {}).items()
        },
    }


def _write_cache(
    cache_dir: Path,
    diffusion_input: ConcentricDiffusionInput,
    system: MultiGroupDiffusionSystem,
    solution: dict[str, Any],
    settings: DiffusionCacheSettings,
    fingerprint: str,
    sph_factors: SphFactorSet | None,
) -> dict[str, Any]:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{fingerprint[:12]}-", dir=cache_dir.parent)
    )
    try:
        operators_dir = temporary / "operators"
        operators_dir.mkdir()
        operator_files = []
        for group, operator in enumerate(system.operators):
            relative = Path("operators") / f"group_{group + 1:03d}.npz"
            sp.save_npz(temporary / relative, operator, compressed=True)
            operator_files.append(str(relative))

        np.savez_compressed(
            temporary / "system_arrays.npz",
            r_edges=system.mesh.r_edges,
            z_edges=system.mesh.z_edges,
            diffusion=system.diffusion,
            absorption=system.absorption,
            nu_fission=system.nu_fission,
            kappa_fission=system.kappa_fission,
            chi=system.chi,
            scatter=system.scatter,
            region_index=system.region_index,
        )
        np.savez_compressed(
            temporary / "clean_solution.npz",
            **_solution_arrays(solution),
        )
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "source": diffusion_input.summary(),
            "settings": settings.as_dict(),
            "sph_factors": (
                None if sph_factors is None else sph_factors.as_dict()
            ),
            "system": {
                "group_count": system.group_count,
                "cell_count": system.cell_count,
                "Nr": system.mesh.nr,
                "Nz": system.mesh.nz,
                "x_insert": system.x_insert,
                "region_labels": list(system.region_labels),
                "operator_files": operator_files,
                "arrays_file": "system_arrays.npz",
            },
            "clean_solution": {
                **_solution_metadata(solution),
                "arrays_file": "clean_solution.npz",
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        temporary.replace(cache_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_cache(
    cache_dir: Path,
    diffusion_input: ConcentricDiffusionInput,
    fingerprint: str,
    settings: DiffusionCacheSettings,
    sph_factors: SphFactorSet | None,
) -> PreparedDiffusionCase:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Diffusion cache schema version does not match")
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("Diffusion cache fingerprint does not match")

    system_metadata = manifest["system"]
    with np.load(cache_dir / system_metadata["arrays_file"]) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mesh = CylindricalMesh2D(
        r_edges=arrays["r_edges"],
        z_edges=arrays["z_edges"],
    )
    operators = tuple(
        sp.load_npz(cache_dir / relative).tocsr()
        for relative in system_metadata["operator_files"]
    )
    if len(operators) != diffusion_input.group_count:
        raise ValueError("Diffusion cache has the wrong number of group matrices")
    expected_matrix_shape = (mesh.cell_count, mesh.cell_count)
    if any(operator.shape != expected_matrix_shape for operator in operators):
        raise ValueError("Diffusion cache contains an invalid group-matrix shape")
    group_count = diffusion_input.group_count
    expected_vector_shape = (mesh.cell_count, group_count)
    for name in (
        "diffusion",
        "absorption",
        "nu_fission",
        "kappa_fission",
        "chi",
    ):
        if arrays[name].shape != expected_vector_shape:
            raise ValueError(
                f"Diffusion cache contains an invalid {name} array shape"
            )
    if arrays["scatter"].shape != (
        mesh.cell_count,
        group_count,
        group_count,
    ):
        raise ValueError("Diffusion cache contains an invalid scatter array shape")
    if arrays["region_index"].shape != (mesh.cell_count,):
        raise ValueError("Diffusion cache contains an invalid region-index shape")

    if sph_factors is None:
        model = diffusion_input.build_model(
            delta_absorption_rod=settings.delta_absorption_rod
        )
    else:
        regions = corrected_regions(
            diffusion_input,
            sph_factors,
            settings.spacing,
        )
        model = diffusion_input.build_model(
            delta_absorption_rod=settings.delta_absorption_rod,
            regions=regions,
        )
    system = MultiGroupDiffusionSystem(
        model=model,
        mesh=mesh,
        operators=operators,
        diffusion=arrays["diffusion"],
        absorption=arrays["absorption"],
        nu_fission=arrays["nu_fission"],
        kappa_fission=arrays["kappa_fission"],
        chi=arrays["chi"],
        scatter=arrays["scatter"],
        region_labels=tuple(system_metadata["region_labels"]),
        region_index=arrays["region_index"],
        x_insert=float(system_metadata["x_insert"]),
    )

    solution_metadata = manifest["clean_solution"]
    with np.load(cache_dir / solution_metadata["arrays_file"]) as archive:
        solution_arrays = {
            name: archive[name].copy() for name in archive.files
        }
    solution = {
        **{
            key: value
            for key, value in solution_metadata.items()
            if key != "arrays_file"
        },
        "phi": solution_arrays["phi"],
        "phi_groups": solution_arrays["phi_groups"],
        "power_density": solution_arrays["power_density"],
        "r_grid": solution_arrays["r_grid"],
        "z_grid": solution_arrays["z_grid"],
        "r_edges": solution_arrays["r_edges"],
        "z_edges": solution_arrays["z_edges"],
        "inner_iterations": solution_arrays["inner_iterations"].tolist(),
    }
    return PreparedDiffusionCase(
        diffusion_input=diffusion_input,
        system=system,
        clean_solution=solution,
        cache_dir=cache_dir,
        cache_hit=True,
        fingerprint=fingerprint,
        manifest=manifest,
        sph_factors=sph_factors,
    )


def prepare_concentric_diffusion_cache(
    diffusion_input: ConcentricDiffusionInput,
    *,
    settings: DiffusionCacheSettings = DiffusionCacheSettings(),
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    force: bool = False,
    sph_factors: SphFactorSet | None = None,
) -> PreparedDiffusionCase:
    if sph_factors is not None and not sph_factors.converged:
        raise ValueError(
            "Cannot prepare a production cache with unconverged SPH factors"
        )
    fingerprint = _fingerprint(diffusion_input, settings, sph_factors)
    cache_dir = Path(cache_root).expanduser().resolve() / fingerprint
    if not force and (cache_dir / "manifest.json").is_file():
        return _load_cache(
            cache_dir,
            diffusion_input,
            fingerprint,
            settings,
            sph_factors,
        )

    if sph_factors is None:
        model = diffusion_input.build_model(
            delta_absorption_rod=settings.delta_absorption_rod
        )
        system = build_multigroup_2d_system(
            model,
            spacing=settings.spacing,
            x_insert=0.0,
        )
    else:
        system = build_sph_corrected_system(
            diffusion_input,
            sph_factors,
            settings.spacing,
        )
    solution = solve_multigroup_system(
        system,
        max_iter=settings.max_iter,
        tol=settings.tol,
        source_tol=settings.source_tol,
        max_inner_iter=settings.max_inner_iter,
        inner_tol=settings.inner_tol,
    )
    manifest = _write_cache(
        cache_dir,
        diffusion_input,
        system,
        solution,
        settings,
        fingerprint,
        sph_factors,
    )
    return PreparedDiffusionCase(
        diffusion_input=diffusion_input,
        system=system,
        clean_solution=solution,
        cache_dir=cache_dir,
        cache_hit=False,
        fingerprint=fingerprint,
        manifest=manifest,
        sph_factors=sph_factors,
    )
