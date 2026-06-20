from __future__ import annotations

import unittest

import numpy as np

from reactor_backend.openmc_mgxs_adapter import PowerMeshReference
from reactor_backend.power_shape import (
    apply_ce_power_shape_correction,
    apply_fixed_power_shape_factor,
)


class PowerShapeCorrectionTests(unittest.TestCase):
    def test_same_mesh_correction_matches_reference_shape(self):
        reference = PowerMeshReference(
            r_edges_cm=[0.0, 1.0, 2.0],
            z_edges_cm=[0.0, 1.0, 2.0],
            mean=[[1.0, 3.0], [2.0, 4.0]],
            std_dev=[[0.0, 0.0], [0.0, 0.0]],
        )
        volumes = np.ones((2, 2))
        diffusion_density = np.ones((2, 2))

        result = apply_ce_power_shape_correction(
            power_density=diffusion_density,
            volumes=volumes,
            r_edges_cm=np.asarray([0.0, 1.0, 2.0]),
            z_edges_cm=np.asarray([0.0, 1.0, 2.0]),
            reference=reference,
        )

        corrected_shape = result.corrected_power_rate / np.sum(
            result.corrected_power_rate
        )
        np.testing.assert_allclose(
            corrected_shape,
            np.asarray(reference.mean) / np.sum(reference.mean),
        )
        self.assertEqual(np.count_nonzero(result.active_bins), 4)
        self.assertAlmostEqual(result.corrected_total, result.diffusion_total)

    def test_fractional_overlap_aggregates_diffusion_to_reference_bins(self):
        reference = PowerMeshReference(
            r_edges_cm=[0.0, 1.0, 2.0],
            z_edges_cm=[0.0, 2.0],
            mean=[[3.0], [1.0]],
            std_dev=[[0.0], [0.0]],
        )
        # One coarse diffusion cell spans both radial reference bins.
        r_edges = np.asarray([0.0, 2.0])
        z_edges = np.asarray([0.0, 2.0])
        volumes = np.asarray([[np.pi * 4.0 * 2.0]])
        diffusion_density = np.asarray([[2.0]])

        result = apply_ce_power_shape_correction(
            power_density=diffusion_density,
            volumes=volumes,
            r_edges_cm=r_edges,
            z_edges_cm=z_edges,
            reference=reference,
        )

        # The single diffusion cell receives the volume-weighted average of the
        # two CE-bin factors: inner radial bin area is 1/4, outer is 3/4.
        expected_factor = 0.25 * 3.0 + 0.75 * (1.0 / 3.0)
        np.testing.assert_allclose(result.correction_factor, [[expected_factor]])

    def test_missing_reference_leaves_power_unchanged(self):
        density = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        volumes = np.ones_like(density)

        result = apply_ce_power_shape_correction(
            power_density=density,
            volumes=volumes,
            r_edges_cm=np.asarray([0.0, 1.0, 2.0]),
            z_edges_cm=np.asarray([0.0, 1.0, 2.0]),
            reference=None,
        )

        np.testing.assert_allclose(result.corrected_power_density, density)
        np.testing.assert_allclose(result.correction_factor, np.ones_like(density))
        self.assertEqual(np.count_nonzero(result.active_bins), 0)

    def test_zero_diffusion_power_returns_zero_correction(self):
        reference = PowerMeshReference(
            r_edges_cm=[0.0, 1.0],
            z_edges_cm=[0.0, 1.0],
            mean=[[1.0]],
            std_dev=[[0.0]],
        )

        result = apply_ce_power_shape_correction(
            power_density=np.asarray([[0.0]]),
            volumes=np.asarray([[1.0]]),
            r_edges_cm=np.asarray([0.0, 1.0]),
            z_edges_cm=np.asarray([0.0, 1.0]),
            reference=reference,
        )

        np.testing.assert_allclose(result.corrected_power_density, [[0.0]])
        self.assertEqual(np.count_nonzero(result.active_bins), 0)
        self.assertEqual(result.corrected_total, 0.0)

    def test_fixed_factor_preserves_corrected_rodded_shape(self):
        density = np.asarray([[2.0, 1.0], [1.0, 4.0]])
        volumes = np.ones_like(density)
        clean_factor = np.asarray([[1.0, 2.0], [0.5, 1.5]])

        result = apply_fixed_power_shape_factor(
            power_density=density,
            volumes=volumes,
            correction_factor=clean_factor,
        )

        expected_rate = density * clean_factor
        np.testing.assert_allclose(result.corrected_power_rate, expected_rate)
        np.testing.assert_allclose(
            result.reference_power_shape,
            expected_rate / np.sum(expected_rate),
        )
        self.assertAlmostEqual(result.corrected_total, float(np.sum(expected_rate)))


if __name__ == "__main__":
    unittest.main()
