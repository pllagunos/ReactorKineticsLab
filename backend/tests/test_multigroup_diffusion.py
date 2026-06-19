from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from reactor_backend.multigroup_diffusion import (
    BOUNDARY_D2O_INTERFACE_VACUUM,
    BOUNDARY_GROUPWISE_FVM,
    ConcentricMeshSpacing,
    CylindricalLayeredModel2D,
    CylindricalRegionZone2D,
    MultiGroupRegion,
    _assemble_global_matrices,
    _fission_density,
    _initial_flux,
    build_concentric_mesh,
    build_multigroup_2d_system,
    solve_multigroup_2d_global_reference,
    solve_multigroup_system,
)


def _region(
    name: str,
    *,
    diffusion: tuple[float, float],
    absorption: tuple[float, float],
    nu_fission: tuple[float, float],
    scatter: tuple[tuple[float, float], tuple[float, float]],
) -> MultiGroupRegion:
    return MultiGroupRegion(
        name=name,
        diffusion=np.asarray(diffusion),
        absorption=np.asarray(absorption),
        nu_fission=np.asarray(nu_fission),
        chi=np.asarray([1.0, 0.0]),
        scatter=np.asarray(scatter),
    )


def _model() -> CylindricalLayeredModel2D:
    fuel = _region(
        "core_fuel_ring_1",
        diffusion=(1.2, 0.35),
        absorption=(0.008, 0.08),
        nu_fission=(0.012, 0.14),
        scatter=((0.15, 0.035), (0.002, 0.22)),
    )
    coolant = _region(
        "core_heavy_water_coolant_and_moderator",
        diffusion=(1.5, 0.5),
        absorption=(0.002, 0.006),
        nu_fission=(0.0, 0.0),
        scatter=((0.18, 0.05), (0.001, 0.28)),
    )
    moderator = _region(
        "moderator",
        diffusion=(1.6, 0.55),
        absorption=(0.001, 0.004),
        nu_fission=(0.0, 0.0),
        scatter=((0.2, 0.06), (0.001, 0.3)),
    )
    reflector = _region(
        "reflector",
        diffusion=(1.4, 0.45),
        absorption=(0.0015, 0.005),
        nu_fission=(0.0, 0.0),
        scatter=((0.16, 0.045), (0.001, 0.25)),
    )
    zones = (
        CylindricalRegionZone2D(
            region=reflector,
            r_max=6.0,
            z_min=-5.0,
            z_max=5.0,
        ),
        CylindricalRegionZone2D(
            region=moderator,
            r_max=5.0,
            z_min=-4.0,
            z_max=4.0,
        ),
        CylindricalRegionZone2D(
            region=coolant,
            r_max=4.0,
            z_min=-3.5,
            z_max=3.5,
        ),
        CylindricalRegionZone2D(
            region=fuel,
            r_min=1.0,
            r_max=3.0,
            z_min=-3.0,
            z_max=3.0,
        ),
    )
    return CylindricalLayeredModel2D(
        core_radius=4.0,
        moderator_radius=5.0,
        reflector_radius=6.0,
        core_height=6.0,
        outer_height=10.0,
        core=fuel,
        moderator=moderator,
        reflector=reflector,
        moderator_height=8.0,
        zones=zones,
    )


class MultiGroupDiffusionTests(unittest.TestCase):
    def test_boundary_fitted_mesh_preserves_resolved_zone_edges(self):
        model = _model()
        mesh = build_concentric_mesh(
            model,
            ConcentricMeshSpacing(
                fuel_radial_cm=0.4,
                core_coolant_radial_cm=0.8,
                moderator_radial_cm=1.5,
                reflector_radial_cm=2.0,
                axial_cm=1.25,
            ),
        )

        for boundary in (1.0, 3.0, 4.0, 5.0, 6.0, model.R_extrap):
            self.assertTrue(np.any(np.isclose(mesh.r_edges, boundary)))
        for boundary in (-5.0, -3.5, -3.0, 3.0, 3.5, 5.0):
            self.assertTrue(np.any(np.isclose(mesh.z_edges, boundary)))

        fuel_widths = np.diff(mesh.r_edges)
        fuel_centers = mesh.r_grid
        self.assertLessEqual(
            np.max(fuel_widths[(fuel_centers >= 1.0) & (fuel_centers <= 3.0)]),
            0.4 + 1.0e-12,
        )

    def test_groupwise_fvm_boundary_uses_physical_reflector_extent(self):
        model = replace(_model(), boundary_condition=BOUNDARY_GROUPWISE_FVM)
        mesh = build_concentric_mesh(
            model,
            ConcentricMeshSpacing(
                fuel_radial_cm=1.0,
                core_coolant_radial_cm=1.0,
                moderator_radial_cm=2.0,
                reflector_radial_cm=2.0,
                axial_cm=2.0,
            ),
        )
        legacy_mesh = build_concentric_mesh(
            _model(),
            ConcentricMeshSpacing(
                fuel_radial_cm=1.0,
                core_coolant_radial_cm=1.0,
                moderator_radial_cm=2.0,
                reflector_radial_cm=2.0,
                axial_cm=2.0,
            ),
        )

        self.assertAlmostEqual(mesh.r_edges[-1], model.reflector_radius)
        self.assertAlmostEqual(mesh.z_edges[0], -0.5 * model.outer_height)
        self.assertAlmostEqual(mesh.z_edges[-1], 0.5 * model.outer_height)
        self.assertLess(mesh.cell_count, legacy_mesh.cell_count)

        system = build_multigroup_2d_system(model, mesh=mesh)
        self.assertEqual(len(system.operators), model.group_count)
        self.assertTrue(
            all(operator.shape == (mesh.cell_count, mesh.cell_count)
                for operator in system.operators)
        )

    def test_d2o_interface_vacuum_truncates_at_d2o_boundary(self):
        model = replace(
            _model(),
            boundary_condition=BOUNDARY_D2O_INTERFACE_VACUUM,
        )
        mesh = build_concentric_mesh(
            model,
            ConcentricMeshSpacing(
                fuel_radial_cm=1.0,
                core_coolant_radial_cm=1.0,
                moderator_radial_cm=2.0,
                reflector_radial_cm=2.0,
                axial_cm=2.0,
            ),
        )

        self.assertAlmostEqual(mesh.r_edges[-1], model.moderator_radius)
        self.assertAlmostEqual(mesh.z_edges[0], -0.5 * model.moderator_height)
        self.assertAlmostEqual(mesh.z_edges[-1], 0.5 * model.moderator_height)

        system = build_multigroup_2d_system(model, mesh=mesh)
        self.assertEqual(system.cell_count, mesh.cell_count)
        self.assertNotIn("reflector", {
            system.region_labels[index]
            for index in np.unique(system.region_index)
        })

    def test_rejects_unknown_boundary_condition(self):
        with self.assertRaisesRegex(ValueError, "boundary_condition"):
            replace(_model(), boundary_condition="not-a-mode")

    def test_matrix_free_fission_source_matches_global_matrix(self):
        system = build_multigroup_2d_system(
            _model(),
            spacing=ConcentricMeshSpacing(
                fuel_radial_cm=1.0,
                core_coolant_radial_cm=1.0,
                moderator_radial_cm=2.0,
                reflector_radial_cm=2.0,
                axial_cm=2.0,
            ),
        )
        phi = np.linspace(
            0.5,
            1.5,
            system.cell_count * system.group_count,
        ).reshape(system.cell_count, system.group_count)
        _, fission = _assemble_global_matrices(system)

        explicit = (fission @ phi.T.reshape(-1)).reshape(
            system.group_count, system.cell_count
        ).T
        matrix_free = system.chi * _fission_density(system, phi)[:, None]

        np.testing.assert_allclose(matrix_free, explicit)

    def test_groupwise_solver_matches_global_reference(self):
        system = build_multigroup_2d_system(
            _model(),
            spacing=ConcentricMeshSpacing(
                fuel_radial_cm=1.0,
                core_coolant_radial_cm=1.0,
                moderator_radial_cm=2.0,
                reflector_radial_cm=2.0,
                axial_cm=2.0,
            ),
        )

        groupwise = solve_multigroup_system(
            system,
            max_iter=300,
            tol=1.0e-9,
            source_tol=1.0e-9,
            max_inner_iter=200,
            inner_tol=1.0e-11,
        )
        global_reference = solve_multigroup_2d_global_reference(
            system,
            max_iter=300,
            tol=1.0e-9,
            source_tol=1.0e-9,
        )

        self.assertTrue(groupwise["converged"])
        self.assertAlmostEqual(
            groupwise["k_eff"],
            global_reference["k_eff"],
            places=7,
        )
        groupwise_shape = groupwise["phi_groups"] / np.max(
            groupwise["phi_groups"]
        )
        global_shape = global_reference["phi_groups"] / np.max(
            global_reference["phi_groups"]
        )
        np.testing.assert_allclose(groupwise_shape, global_shape, rtol=2.0e-6)

    def test_native_cell_group_initial_flux_preserves_ordering(self):
        system = build_multigroup_2d_system(
            _model(),
            spacing=ConcentricMeshSpacing(
                fuel_radial_cm=1.0,
                core_coolant_radial_cm=1.0,
                moderator_radial_cm=2.0,
                reflector_radial_cm=2.0,
                axial_cm=2.0,
            ),
        )
        phi0 = np.arange(
            system.cell_count * system.group_count,
            dtype=float,
        ).reshape(system.cell_count, system.group_count) + 1.0

        np.testing.assert_array_equal(_initial_flux(system, phi0), phi0)


if __name__ == "__main__":
    unittest.main()
