"""Post-solve power-shape correction against an OpenMC kappa-fission mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .openmc_mgxs_adapter import PowerMeshReference


@dataclass(frozen=True)
class PowerShapeCorrection:
    corrected_power_density: np.ndarray
    corrected_power_rate: np.ndarray
    correction_factor: np.ndarray
    reference_power_shape: np.ndarray
    diffusion_power_shape: np.ndarray
    active_bins: np.ndarray
    reference_total: float
    diffusion_total: float
    corrected_total: float


def _overlap_fraction_matrix(
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    *,
    radial: bool,
) -> np.ndarray:
    source_edges = np.asarray(source_edges, dtype=float).reshape(-1)
    target_edges = np.asarray(target_edges, dtype=float).reshape(-1)
    fractions = np.zeros(
        (target_edges.size - 1, source_edges.size - 1),
        dtype=float,
    )
    for target_index, (target_lo, target_hi) in enumerate(
        zip(target_edges[:-1], target_edges[1:])
    ):
        lo = np.maximum(target_lo, source_edges[:-1])
        hi = np.minimum(target_hi, source_edges[1:])
        overlap = hi > lo
        if not np.any(overlap):
            continue
        if radial:
            numerator = hi[overlap] ** 2 - lo[overlap] ** 2
            denominator = (
                source_edges[1:][overlap] ** 2
                - source_edges[:-1][overlap] ** 2
            )
        else:
            numerator = hi[overlap] - lo[overlap]
            denominator = source_edges[1:][overlap] - source_edges[:-1][overlap]
        fractions[target_index, np.flatnonzero(overlap)] = numerator / denominator
    return fractions


def apply_ce_power_shape_correction(
    *,
    power_density: np.ndarray,
    volumes: np.ndarray,
    r_edges_cm: np.ndarray,
    z_edges_cm: np.ndarray,
    reference: PowerMeshReference | None,
    relative_floor: float = 1.0e-14,
) -> PowerShapeCorrection:
    """Apply a CE/diffusion power-shape ratio as a post-solve correction.

    The correction is shape-only. The CE kappa-fission mesh and diffusion
    power mapped onto that mesh are both normalized by their totals before
    forming ``C = P_CE / P_diff``.
    """

    raw_density = np.asarray(power_density, dtype=float)
    volumes_2d = np.asarray(volumes, dtype=float)
    if raw_density.shape != volumes_2d.shape:
        raise ValueError(
            "power_density and volumes must have the same (r, z) shape"
        )
    if reference is None:
        raw_rate = raw_density * volumes_2d
        shape = raw_density.shape
        return PowerShapeCorrection(
            corrected_power_density=raw_density.copy(),
            corrected_power_rate=raw_rate,
            correction_factor=np.ones(shape, dtype=float),
            reference_power_shape=np.zeros(shape, dtype=float),
            diffusion_power_shape=np.zeros(shape, dtype=float),
            active_bins=np.zeros(shape, dtype=bool),
            reference_total=0.0,
            diffusion_total=float(np.sum(raw_rate)),
            corrected_total=float(np.sum(raw_rate)),
        )

    r_edges = np.asarray(r_edges_cm, dtype=float).reshape(-1)
    z_edges = np.asarray(z_edges_cm, dtype=float).reshape(-1)
    expected_shape = (r_edges.size - 1, z_edges.size - 1)
    if raw_density.shape != expected_shape:
        raise ValueError(
            "power_density shape does not match diffusion mesh edges: "
            f"expected {expected_shape}, got {raw_density.shape}"
        )

    reference_rate = np.asarray(reference.mean, dtype=float)
    if reference_rate.shape != (
        reference.r_edges_cm.size - 1,
        reference.z_edges_cm.size - 1,
    ):
        raise ValueError("reference power mesh shape does not match its edges")
    if np.any(reference_rate < 0.0):
        raise ValueError("reference power mesh must not contain negative power")

    raw_rate = raw_density * volumes_2d
    r_to_reference = _overlap_fraction_matrix(
        r_edges,
        reference.r_edges_cm,
        radial=True,
    )
    z_to_reference = _overlap_fraction_matrix(
        z_edges,
        reference.z_edges_cm,
        radial=False,
    )
    diffusion_on_reference = r_to_reference @ raw_rate @ z_to_reference.T

    reference_total = float(np.sum(reference_rate))
    diffusion_total = float(np.sum(diffusion_on_reference))
    if (
        not np.isfinite(reference_total)
        or reference_total <= 0.0
        or not np.isfinite(diffusion_total)
        or diffusion_total <= 0.0
    ):
        return PowerShapeCorrection(
            corrected_power_density=np.zeros_like(raw_density),
            corrected_power_rate=np.zeros_like(raw_rate),
            correction_factor=np.zeros_like(raw_density),
            reference_power_shape=np.zeros_like(reference_rate),
            diffusion_power_shape=np.zeros_like(reference_rate),
            active_bins=np.zeros_like(reference_rate, dtype=bool),
            reference_total=reference_total,
            diffusion_total=diffusion_total,
            corrected_total=0.0,
        )

    reference_shape = reference_rate / reference_total
    diffusion_shape = diffusion_on_reference / diffusion_total
    floor = max(0.0, float(relative_floor))
    active = (reference_shape > floor) & (diffusion_shape > floor)

    factor_on_reference = np.zeros_like(reference_shape)
    np.divide(
        reference_shape,
        diffusion_shape,
        out=factor_on_reference,
        where=active,
    )

    factor_on_diffusion = r_to_reference.T @ factor_on_reference @ z_to_reference
    corrected_rate = raw_rate * factor_on_diffusion
    corrected_density = np.zeros_like(raw_density)
    np.divide(
        corrected_rate,
        volumes_2d,
        out=corrected_density,
        where=volumes_2d > 0.0,
    )

    return PowerShapeCorrection(
        corrected_power_density=corrected_density,
        corrected_power_rate=corrected_rate,
        correction_factor=factor_on_diffusion,
        reference_power_shape=reference_shape,
        diffusion_power_shape=diffusion_shape,
        active_bins=active,
        reference_total=reference_total,
        diffusion_total=diffusion_total,
        corrected_total=float(np.sum(corrected_rate)),
    )


def apply_fixed_power_shape_factor(
    *,
    power_density: np.ndarray,
    volumes: np.ndarray,
    correction_factor: np.ndarray,
) -> PowerShapeCorrection:
    """Apply a precomputed clean-core shape factor to a new power map.

    This is the rod-dependent display path: the CE/diffusion factor is fixed
    from the clean state, while the relative shape perturbation comes from the
    rodded diffusion solve.
    """

    raw_density = np.asarray(power_density, dtype=float)
    volumes_2d = np.asarray(volumes, dtype=float)
    factor = np.asarray(correction_factor, dtype=float)
    if raw_density.shape != volumes_2d.shape:
        raise ValueError(
            "power_density and volumes must have the same (r, z) shape"
        )
    if factor.shape != raw_density.shape:
        raise ValueError("correction_factor must match power_density shape")
    if np.any(~np.isfinite(factor)) or np.any(factor < 0.0):
        raise ValueError("correction_factor must be finite and non-negative")

    raw_rate = raw_density * volumes_2d
    corrected_rate = raw_rate * factor
    corrected_density = np.zeros_like(raw_density)
    np.divide(
        corrected_rate,
        volumes_2d,
        out=corrected_density,
        where=volumes_2d > 0.0,
    )
    total = float(np.sum(raw_rate))
    corrected_total = float(np.sum(corrected_rate))
    diffusion_shape = (
        raw_rate / total if np.isfinite(total) and total > 0.0 else np.zeros_like(raw_rate)
    )
    corrected_shape = (
        corrected_rate / corrected_total
        if np.isfinite(corrected_total) and corrected_total > 0.0
        else np.zeros_like(corrected_rate)
    )
    return PowerShapeCorrection(
        corrected_power_density=corrected_density,
        corrected_power_rate=corrected_rate,
        correction_factor=factor.copy(),
        reference_power_shape=corrected_shape,
        diffusion_power_shape=diffusion_shape,
        active_bins=factor > 0.0,
        reference_total=corrected_total,
        diffusion_total=total,
        corrected_total=corrected_total,
    )
