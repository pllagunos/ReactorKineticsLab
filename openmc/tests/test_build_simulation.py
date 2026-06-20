from __future__ import annotations

import math
import unittest

import numpy as np
import openmc

from build_simulation import _build_axial_mesh
from ploting import _mesh_bin_density, _mesh_display_plane


class BuildSimulationTests(unittest.TestCase):
    def test_axial_tally_mesh_covers_full_cylindrical_reactor(self):
        mesh = _build_axial_mesh(
            mesh_shape=(16, 20),
            radius_cm=500.0,
            half_height_cm=500.0,
        )

        self.assertIsInstance(mesh, openmc.CylindricalMesh)
        self.assertEqual(tuple(mesh.dimension), (16, 1, 20))
        np.testing.assert_allclose(mesh.r_grid[[0, -1]], [0.0, 500.0])
        np.testing.assert_allclose(
            mesh.phi_grid,
            [0.0, 2.0 * math.pi],
        )
        np.testing.assert_allclose(
            mesh.z_grid[[0, -1]],
            [-500.0, 500.0],
        )

    def test_cylindrical_plot_density_removes_annulus_volume_bias(self):
        mesh = _build_axial_mesh(
            mesh_shape=(4, 2),
            radius_cm=40.0,
            half_height_cm=10.0,
        )
        volumes = np.asarray(mesh.volumes).reshape(tuple(mesh.dimension))
        density = _mesh_bin_density(volumes.squeeze(), mesh)

        np.testing.assert_allclose(density, 1.0)
        mirrored = _mesh_display_plane(density, mesh)
        self.assertEqual(mirrored.shape, (2, 8))
        np.testing.assert_allclose(mirrored, 1.0)

    def test_cylindrical_plot_mirrors_radial_bins_about_centerline(self):
        mesh = _build_axial_mesh(
            mesh_shape=(3, 2),
            radius_cm=30.0,
            half_height_cm=10.0,
        )
        radial_axial = np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )

        displayed = _mesh_display_plane(radial_axial, mesh)

        np.testing.assert_allclose(
            displayed,
            [
                [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                [30.0, 20.0, 10.0, 10.0, 20.0, 30.0],
            ],
        )


if __name__ == "__main__":
    unittest.main()
