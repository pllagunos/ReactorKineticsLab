from __future__ import annotations

import copy
import unittest
from dataclasses import replace

import numpy as np

from reactor_backend.multigroup_diffusion import (
    ConcentricMeshSpacing,
    build_multigroup_2d_system,
    solve_multigroup_system,
)
from reactor_backend.multigroup_sph import (
    REFERENCE_MODE_AXIAL_REGION_FLUX,
    REFERENCE_MODE_AXIAL_POWER_SHAPE,
    SphConvergenceError,
    SphFactorSet,
    SphSettings,
    axialized_diffusion_input,
    build_axial_flux_sph_reference,
    build_axial_power_sph_reference,
    build_sph_corrected_system,
    build_sph_reference,
    corrected_regions,
    evaluate_sph_qualification,
    fit_sph_factors,
    qualify_mesh,
    region_integrated_flux,
    sph_source_fingerprint,
)
from reactor_backend.openmc_mgxs_adapter import (
    ContinuousEnergyReference,
    PowerMeshReference,
    ReferenceValues,
    load_concentric_diffusion_input,
)

try:
    from .test_openmc_mgxs_adapter import AdapterFixture, _export
except ImportError:
    from test_openmc_mgxs_adapter import AdapterFixture, _export


class MultiGroupSphTests(unittest.TestCase):
    def setUp(self):
        self.fixture = AdapterFixture(copy.deepcopy(_export()))
        self.addCleanup(self.fixture.close)
        self.diffusion_input = load_concentric_diffusion_input(
            self.fixture.path
        )
        self.spacing = ConcentricMeshSpacing(
            fuel_radial_cm=1.0,
            core_coolant_radial_cm=50.0,
            moderator_radial_cm=100.0,
            reflector_radial_cm=250.0,
            axial_cm=250.0,
        )

    def _unit_factors(self) -> SphFactorSet:
        labels = tuple(
            dict.fromkeys(
                zone.region.name for zone in self.diffusion_input.zones
            )
        )
        factors = np.ones(
            (len(labels), self.diffusion_input.group_count),
            dtype=float,
        )
        return SphFactorSet(
            region_labels=labels,
            factors=factors,
            active=np.ones_like(factors, dtype=bool),
            converged=True,
            iterations=0,
            history=(),
            source_fingerprint=sph_source_fingerprint(
                self.diffusion_input,
                self.spacing,
            ),
            mesh_spacing=self.spacing.as_dict(),
            provisional=True,
        )

    def _diffusion_reference_input(self):
        system = build_multigroup_2d_system(
            self.diffusion_input.build_model(),
            spacing=self.spacing,
        )
        solution = solve_multigroup_system(
            system,
            max_iter=300,
            tol=1.0e-6,
            source_tol=1.0e-3,
            max_inner_iter=200,
            inner_tol=1.0e-4,
        )
        labels = tuple(
            dict.fromkeys(
                zone.region.name for zone in self.diffusion_input.zones
            )
        )
        integrated = region_integrated_flux(
            system,
            solution,
            region_labels=labels,
        )
        reference = ContinuousEnergyReference(
            energy_order="fast-to-thermal",
            normalization="synthetic diffusion reference",
            region_flux={
                label: ReferenceValues(
                    mean=integrated[index],
                    std_dev=np.zeros(self.diffusion_input.group_count),
                )
                for index, label in enumerate(labels)
            },
            master_flux=ReferenceValues(
                mean=np.sum(integrated, axis=0),
                std_dev=np.zeros(self.diffusion_input.group_count),
            ),
            power_mesh=None,
        )
        return replace(
            self.diffusion_input,
            ce_reference=reference,
            openmc_reference={
                "keff": float(solution["k_eff"]),
                "keff_std_dev": 1.0e-4,
                "reactivity_pcm": 0.0,
            },
        )

    def _diffusion_reference_input_with_power(self):
        reference_input = self._diffusion_reference_input()
        system = build_multigroup_2d_system(
            reference_input.build_model(),
            spacing=self.spacing,
        )
        solution = solve_multigroup_system(
            system,
            max_iter=300,
            tol=1.0e-6,
            source_tol=1.0e-3,
            max_inner_iter=200,
            inner_tol=1.0e-4,
        )
        assert reference_input.ce_reference is not None
        power_rate = (
            solution["power_density"]
            * system.mesh.volumes.reshape(system.mesh.nr, system.mesh.nz)
        )
        reference = replace(
            reference_input.ce_reference,
            power_mesh=PowerMeshReference(
                r_edges_cm=system.mesh.r_edges,
                z_edges_cm=system.mesh.z_edges,
                mean=power_rate,
                std_dev=np.zeros_like(power_rate),
            ),
        )
        return (
            replace(
                reference_input,
                ce_reference=reference,
                openmc_reference={
                    "keff": float(solution["k_eff"] + 0.05),
                    "keff_std_dev": 5.0e-4,
                    "reactivity_pcm": 0.0,
                },
            ),
            system,
            solution,
        )

    def _axial_diffusion_reference_input(self, axial_zones):
        axial_input = axialized_diffusion_input(
            self.diffusion_input,
            axial_zones,
        )
        system = build_multigroup_2d_system(
            axial_input.build_model(),
            spacing=self.spacing,
        )
        solution = solve_multigroup_system(
            system,
            max_iter=300,
            tol=1.0e-6,
            source_tol=1.0e-3,
            max_inner_iter=200,
            inner_tol=1.0e-4,
        )
        labels = tuple(
            dict.fromkeys(zone.region.name for zone in axial_input.zones)
        )
        integrated = region_integrated_flux(
            system,
            solution,
            region_labels=labels,
        )
        region_flux = {}
        axial_region_flux = {}
        for index, label in enumerate(labels):
            values = ReferenceValues(
                mean=integrated[index],
                std_dev=np.zeros(self.diffusion_input.group_count),
            )
            if "__axial_" in label:
                axial_region_flux[label] = values
            else:
                region_flux[label] = values
        reference = ContinuousEnergyReference(
            energy_order="fast-to-thermal",
            normalization="synthetic axial diffusion reference",
            region_flux=region_flux,
            master_flux=ReferenceValues(
                mean=np.sum(integrated, axis=0),
                std_dev=np.zeros(self.diffusion_input.group_count),
            ),
            power_mesh=None,
            axial_region_flux=axial_region_flux,
        )
        return (
            replace(
                self.diffusion_input,
                ce_reference=reference,
                openmc_reference={
                    "keff": float(solution["k_eff"]),
                    "keff_std_dev": 1.0e-4,
                    "reactivity_pcm": 0.0,
                },
            ),
            system,
            solution,
        )

    def test_unit_factors_preserve_all_region_constants(self):
        factors = self._unit_factors()
        corrected = corrected_regions(
            self.diffusion_input,
            factors,
            self.spacing,
        )

        for label, original in self.diffusion_input.regions.items():
            np.testing.assert_array_equal(
                corrected[label].diffusion,
                original.diffusion,
            )
            np.testing.assert_array_equal(
                corrected[label].absorption,
                original.absorption,
            )
            np.testing.assert_array_equal(
                corrected[label].nu_fission,
                original.nu_fission,
            )
            np.testing.assert_array_equal(
                corrected[label].kappa_fission,
                original.kappa_fission,
            )
            np.testing.assert_array_equal(
                corrected[label].scatter,
                original.scatter,
            )
            np.testing.assert_array_equal(
                corrected[label].chi,
                original.chi,
            )

        with self.assertRaises(ValueError):
            factors.factors[0, 0] = 2.0
        with self.assertRaises(TypeError):
            factors.mesh_spacing["axial_cm"] = 1.0

    def test_unit_factors_reproduce_uncorrected_solve_deterministically(self):
        baseline_system = build_multigroup_2d_system(
            self.diffusion_input.build_model(),
            spacing=self.spacing,
        )
        corrected_system = build_sph_corrected_system(
            self.diffusion_input,
            self._unit_factors(),
            self.spacing,
        )
        for baseline, corrected in zip(
            baseline_system.operators,
            corrected_system.operators,
            strict=True,
        ):
            np.testing.assert_array_equal(
                baseline.toarray(),
                corrected.toarray(),
            )

        solve_options = {
            "max_iter": 300,
            "tol": 1.0e-6,
            "source_tol": 1.0e-3,
            "max_inner_iter": 200,
            "inner_tol": 1.0e-4,
        }
        baseline_solution = solve_multigroup_system(
            baseline_system,
            **solve_options,
        )
        first = solve_multigroup_system(corrected_system, **solve_options)
        second = solve_multigroup_system(corrected_system, **solve_options)

        self.assertAlmostEqual(
            baseline_solution["k_eff"],
            first["k_eff"],
            places=12,
        )
        self.assertLess(
            abs(second["k_eff"] - first["k_eff"]) * 1.0e5,
            1.0,
        )
        np.testing.assert_allclose(
            baseline_solution["phi_groups"],
            first["phi_groups"],
            rtol=0.0,
            atol=0.0,
        )

    def test_factors_scale_transport_and_reaction_terms_consistently(self):
        factors = self._unit_factors()
        scaled = replace(factors, factors=np.full_like(factors.factors, 2.0))

        corrected = corrected_regions(
            self.diffusion_input,
            scaled,
            self.spacing,
        )
        original = self.diffusion_input.regions["core_fuel_ring_1"]
        region = corrected["core_fuel_ring_1"]

        np.testing.assert_allclose(region.diffusion, original.diffusion / 2.0)
        np.testing.assert_allclose(region.absorption, original.absorption * 2.0)
        np.testing.assert_allclose(region.nu_fission, original.nu_fission * 2.0)
        np.testing.assert_allclose(
            region.kappa_fission,
            original.kappa_fission * 2.0,
        )
        np.testing.assert_allclose(region.scatter, original.scatter * 2.0)
        np.testing.assert_array_equal(region.chi, original.chi)

    def test_axial_unit_factors_preserve_axialized_system(self):
        axial_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }
        axial_input = axialized_diffusion_input(
            self.diffusion_input,
            axial_zones,
        )
        labels = tuple(
            dict.fromkeys(zone.region.name for zone in axial_input.zones)
        )
        self.assertIn("core_fuel_ring_1__axial_lower", labels)
        self.assertIn("core_fuel_ring_1__axial_upper", labels)

        shape = (len(labels), self.diffusion_input.group_count)
        factors = SphFactorSet(
            region_labels=labels,
            factors=np.ones(shape),
            active=np.ones(shape, dtype=bool),
            converged=True,
            iterations=0,
            history=(),
            source_fingerprint=sph_source_fingerprint(
                self.diffusion_input,
                self.spacing,
                reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
                axial_sph_zones=axial_zones,
            ),
            mesh_spacing=self.spacing.as_dict(),
            provisional=False,
            reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
            axial_sph_zones=axial_zones,
        )

        baseline = build_multigroup_2d_system(
            axial_input.build_model(),
            spacing=self.spacing,
        )
        corrected = build_sph_corrected_system(
            self.diffusion_input,
            factors,
            self.spacing,
        )

        self.assertTrue(np.any(np.isclose(corrected.mesh.z_edges, 0.0)))
        np.testing.assert_array_equal(corrected.diffusion, baseline.diffusion)
        np.testing.assert_array_equal(corrected.absorption, baseline.absorption)
        np.testing.assert_array_equal(
            corrected.kappa_fission,
            baseline.kappa_fission,
        )
        for baseline_operator, corrected_operator in zip(
            baseline.operators,
            corrected.operators,
            strict=True,
        ):
            np.testing.assert_array_equal(
                corrected_operator.toarray(),
                baseline_operator.toarray(),
            )

    def test_axial_power_reference_allocates_fractional_power_mesh_bins(self):
        assert self.diffusion_input.ce_reference is not None
        reference = replace(
            self.diffusion_input.ce_reference,
            power_mesh=PowerMeshReference(
                r_edges_cm=np.asarray([4.5, 5.0]),
                z_edges_cm=np.asarray([-150.0, 0.0, 150.0]),
                mean=np.asarray([[2.0, 4.0]]),
                std_dev=np.zeros((1, 2)),
            ),
        )
        axial_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 75.0),
                ("upper", 75.0, 150.0),
            )
        }
        result = build_axial_power_sph_reference(
            replace(self.diffusion_input, ce_reference=reference),
            axial_sph_zones=axial_zones,
            region_labels=(
                "core_fuel_ring_1__axial_lower",
                "core_fuel_ring_1__axial_upper",
            ),
        )

        np.testing.assert_allclose(result.power, [2.0 / 3.0, 1.0 / 3.0])
        self.assertTrue(np.all(result.active))

    def test_power_shape_sph_keeps_non_fissile_axial_clones_inactive(self):
        central = "core_central_moderator_channel"
        regions = dict(self.diffusion_input.regions)
        regions[central] = replace(
            regions[central],
            nu_fission=np.zeros_like(regions[central].nu_fission),
            kappa_fission=np.zeros_like(regions[central].kappa_fission),
        )
        reference_input = replace(self.diffusion_input, regions=regions)
        axial_zones = {
            central: (
                ("lower_extension", -200.0, -150.0),
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            ),
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            ),
        }

        reference = build_axial_power_sph_reference(
            reference_input,
            SphSettings(maximum_relative_std_dev=1.0),
            axial_sph_zones=axial_zones,
        )

        central_active = [
            bool(reference.active[index])
            for index, label in enumerate(reference.region_labels)
            if label.startswith(f"{central}__axial_")
        ]
        fuel_active = [
            bool(reference.active[index])
            for index, label in enumerate(reference.region_labels)
            if label.startswith("core_fuel_ring_1__axial_")
        ]
        self.assertTrue(central_active)
        self.assertFalse(any(central_active))
        self.assertTrue(any(fuel_active))

    def test_axial_flux_reference_uses_clone_flux_and_base_exclusions(self):
        axial_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }
        reference_input, _, _ = self._axial_diffusion_reference_input(
            axial_zones
        )

        reference = build_axial_flux_sph_reference(
            reference_input,
            SphSettings(
                maximum_relative_std_dev=1.0,
                excluded_region_labels=("core_fuel_ring_1",),
            ),
            axial_sph_zones=axial_zones,
        )

        lower = reference.region_labels.index(
            "core_fuel_ring_1__axial_lower"
        )
        upper = reference.region_labels.index(
            "core_fuel_ring_1__axial_upper"
        )
        self.assertFalse(np.any(reference.active[lower]))
        self.assertFalse(np.any(reference.active[upper]))
        self.assertGreater(np.sum(reference.flux[lower]), 0.0)
        self.assertGreater(np.sum(reference.flux[upper]), 0.0)

    def test_axial_flux_reference_requires_axial_clone_flux(self):
        axial_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }

        with self.assertRaisesRegex(ValueError, "missing axial"):
            build_axial_flux_sph_reference(
                self.diffusion_input,
                SphSettings(maximum_relative_std_dev=1.0),
                axial_sph_zones=axial_zones,
            )

    def test_axial_flux_reference_can_activate_non_fissile_moderator(self):
        central = "core_central_moderator_channel"
        axial_zones = {
            central: (
                ("lower_extension", -200.0, -150.0),
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }
        reference_input, _, _ = self._axial_diffusion_reference_input(
            axial_zones
        )

        reference = build_axial_flux_sph_reference(
            reference_input,
            SphSettings(
                maximum_relative_std_dev=1.0,
                excluded_region_labels=("reflector", "core_control_rod"),
            ),
            axial_sph_zones=axial_zones,
        )

        central_active = [
            bool(np.any(reference.active[index]))
            for index, label in enumerate(reference.region_labels)
            if label.startswith(f"{central}__axial_")
        ]
        self.assertEqual(len(central_active), 3)
        self.assertTrue(all(central_active))

    def test_axial_region_flux_fingerprint_includes_reference_values(self):
        axial_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }
        reference_input, _, _ = self._axial_diffusion_reference_input(
            axial_zones
        )
        changed_axial_flux = dict(
            reference_input.ce_reference.axial_region_flux
        )
        changed_axial_flux["core_fuel_ring_1__axial_lower"] = ReferenceValues(
            mean=(
                changed_axial_flux[
                    "core_fuel_ring_1__axial_lower"
                ].mean
                * 1.01
            ),
            std_dev=np.zeros(self.diffusion_input.group_count),
        )
        changed_reference = replace(
            reference_input.ce_reference,
            axial_region_flux=changed_axial_flux,
        )

        first = sph_source_fingerprint(
            reference_input,
            self.spacing,
            reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
            axial_sph_zones=axial_zones,
        )
        second = sph_source_fingerprint(
            replace(reference_input, ce_reference=changed_reference),
            self.spacing,
            reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
            axial_sph_zones=axial_zones,
        )

        self.assertNotEqual(first, second)

    def test_axial_region_flux_fit_finds_unit_factors_for_matching_reference(self):
        axial_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }
        reference_input, _, _ = self._axial_diffusion_reference_input(
            axial_zones
        )

        result = fit_sph_factors(
            reference_input,
            spacing=self.spacing,
            settings=SphSettings(
                max_iterations=6,
                stable_iterations=2,
                flux_tolerance=1.0e-4,
                k_stability_pcm=0.1,
                maximum_relative_std_dev=1.0,
            ),
            reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
            axial_sph_zones=axial_zones,
        )

        self.assertTrue(result.factors.converged)
        self.assertEqual(
            result.factors.reference_mode,
            REFERENCE_MODE_AXIAL_REGION_FLUX,
        )
        np.testing.assert_allclose(
            result.factors.factors,
            1.0,
            rtol=1.0e-6,
            atol=1.0e-8,
        )

    def test_axial_region_flux_factor_set_round_trips(self):
        axial_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }
        reference_input, _, _ = self._axial_diffusion_reference_input(
            axial_zones
        )
        axial_input = axialized_diffusion_input(reference_input, axial_zones)
        labels = tuple(
            dict.fromkeys(zone.region.name for zone in axial_input.zones)
        )
        shape = (len(labels), reference_input.group_count)
        factors = SphFactorSet(
            region_labels=labels,
            factors=np.ones(shape),
            active=np.ones(shape, dtype=bool),
            converged=True,
            iterations=2,
            history=(),
            source_fingerprint=sph_source_fingerprint(
                reference_input,
                self.spacing,
                reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
                axial_sph_zones=axial_zones,
            ),
            mesh_spacing=self.spacing.as_dict(),
            provisional=False,
            reference_mode=REFERENCE_MODE_AXIAL_REGION_FLUX,
            axial_sph_zones=axial_zones,
        )

        reloaded = SphFactorSet.from_dict(factors.as_dict())

        self.assertEqual(
            reloaded.reference_mode,
            REFERENCE_MODE_AXIAL_REGION_FLUX,
        )
        self.assertEqual(reloaded.axial_sph_zones, factors.axial_sph_zones)
        np.testing.assert_array_equal(reloaded.factors, factors.factors)

    def test_axial_sph_fingerprint_includes_mode_and_zone_boundaries(self):
        first_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }
        second_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 75.0),
                ("upper", 75.0, 150.0),
            )
        }

        base = sph_source_fingerprint(self.diffusion_input, self.spacing)
        first = sph_source_fingerprint(
            self.diffusion_input,
            self.spacing,
            reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
            axial_sph_zones=first_zones,
        )
        second = sph_source_fingerprint(
            self.diffusion_input,
            self.spacing,
            reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
            axial_sph_zones=second_zones,
        )

        self.assertNotEqual(base, first)
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, "requires axial_sph_zones"):
            sph_source_fingerprint(
                self.diffusion_input,
                self.spacing,
                reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
            )

    def test_axial_power_fit_runs_fixed_point_iteration(self):
        axial_zones = {
            "core_fuel_ring_1": (
                ("lower", -150.0, 0.0),
                ("upper", 0.0, 150.0),
            )
        }

        with self.assertRaisesRegex(SphConvergenceError, "did not converge") as raised:
            fit_sph_factors(
                self.diffusion_input,
                spacing=self.spacing,
                settings=SphSettings(
                    max_iterations=1,
                    stable_iterations=2,
                    maximum_relative_std_dev=1.0,
                ),
                reference_mode=REFERENCE_MODE_AXIAL_POWER_SHAPE,
                axial_sph_zones=axial_zones,
            )

        result = raised.exception.result
        self.assertEqual(
            result.factors.reference_mode,
            REFERENCE_MODE_AXIAL_POWER_SHAPE,
        )
        self.assertIn(
            "core_fuel_ring_1__axial_lower",
            result.factors.region_labels,
        )
        self.assertIn("maximum_axial_zone_power_error", result.qualification)

    def test_rejects_stale_factors_for_different_mesh(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            corrected_regions(
                self.diffusion_input,
                self._unit_factors(),
                replace(self.spacing, axial_cm=200.0),
            )

    def test_fits_unit_factors_to_a_matching_reference(self):
        result = fit_sph_factors(
            self._diffusion_reference_input(),
            spacing=self.spacing,
            settings=SphSettings(
                max_iterations=6,
                stable_iterations=2,
                flux_tolerance=1.0e-4,
                k_stability_pcm=0.1,
            ),
        )

        self.assertTrue(result.factors.converged)
        np.testing.assert_allclose(
            result.factors.factors,
            1.0,
            rtol=1.0e-6,
            atol=1.0e-8,
        )

    def test_reports_non_convergence(self):
        with self.assertRaisesRegex(
            SphConvergenceError,
            "did not converge",
        ) as raised:
            fit_sph_factors(
                self._diffusion_reference_input(),
                spacing=self.spacing,
                settings=SphSettings(
                    max_iterations=1,
                    stable_iterations=2,
                ),
            )
        result = raised.exception.result
        self.assertFalse(result.factors.converged)
        self.assertEqual(len(result.factor_history), 1)
        self.assertEqual(result.factors.iterations, 1)
        self.assertFalse(result.qualification["qualified"])

    def test_qualification_reports_but_does_not_gate_on_k_error(self):
        reference_input, system, solution = (
            self._diffusion_reference_input_with_power()
        )
        reference = build_sph_reference(reference_input)
        factors = SphFactorSet(
            region_labels=reference.region_labels,
            factors=np.ones_like(reference.flux),
            active=reference.active,
            converged=True,
            iterations=2,
            history=(
                {
                    "iteration": 2,
                    "k_eff": float(solution["k_eff"]),
                    "k_change_pcm": 0.0,
                    "max_flux_error": 0.0,
                    "max_factor_log_change": 0.0,
                    "minimum_factor": 1.0,
                    "maximum_factor": 1.0,
                },
            ),
            source_fingerprint=sph_source_fingerprint(
                reference_input,
                self.spacing,
            ),
            mesh_spacing=self.spacing.as_dict(),
            provisional=False,
        )

        report = evaluate_sph_qualification(
            reference_input,
            system,
            solution,
            factors,
        )

        self.assertTrue(report["qualified"])
        self.assertGreater(abs(report["k_error_pcm"]), 1000.0)
        self.assertNotIn("k_pcm", report["thresholds"])
        self.assertNotIn("reported_k_pcm", report["thresholds"])
        self.assertNotIn("reference_keff_std_dev_pcm", report["thresholds"])
        self.assertNotIn("reported_power_maximum", report["thresholds"])
        self.assertNotIn("solve_time_s", report["thresholds"])
        self.assertEqual(
            set(report["thresholds"]),
            {"region_group_flux", "radial_power_rms", "axial_power_rms"},
        )
        self.assertEqual(
            set(report["acceptance_criteria"]),
            {"region_group_flux", "radial_power_rms", "axial_power_rms"},
        )

    def test_rejects_reference_without_statistically_active_flux(self):
        reference_input = self._diffusion_reference_input()
        assert reference_input.ce_reference is not None
        inactive_reference = replace(
            reference_input.ce_reference,
            region_flux={
                label: ReferenceValues(
                    mean=values.mean,
                    std_dev=np.maximum(values.mean, 1.0),
                )
                for label, values in (
                    reference_input.ce_reference.region_flux.items()
                )
            },
        )

        with self.assertRaisesRegex(ValueError, "no statistically active"):
            fit_sph_factors(
                replace(reference_input, ce_reference=inactive_reference),
                spacing=self.spacing,
                settings=SphSettings(maximum_relative_std_dev=0.5),
            )

    def test_builds_normalized_reference_with_uncertainty_mask(self):
        reference = build_sph_reference(
            self.diffusion_input,
            SphSettings(maximum_relative_std_dev=0.015),
        )

        self.assertEqual(
            reference.flux.shape,
            (
                len(reference.region_labels),
                self.diffusion_input.group_count,
            ),
        )
        self.assertTrue(np.all(reference.flux >= 0.0))
        self.assertTrue(np.any(reference.active))
        self.assertFalse(reference.flux.flags.writeable)
        self.assertFalse(reference.active.flags.writeable)
        for label in ("reflector", "core_control_rod"):
            if label in reference.region_labels:
                row = reference.region_labels.index(label)
                self.assertFalse(np.any(reference.active[row]))

    def test_rejects_zero_reference_fission_production(self):
        reference_input = self._diffusion_reference_input()
        zero_fission_regions = {
            label: replace(
                region,
                nu_fission=np.zeros_like(region.nu_fission),
            )
            for label, region in reference_input.regions.items()
        }
        reference_input = replace(
            reference_input,
            regions=zero_fission_regions,
        )

        with self.assertRaisesRegex(ValueError, "zero fission production"):
            build_sph_reference(reference_input)

    def test_identical_mesh_qualifies(self):
        report = qualify_mesh(
            self.diffusion_input,
            reference_spacing=self.spacing,
            candidate_spacing=self.spacing,
        )

        self.assertTrue(report["accepted"])
        self.assertAlmostEqual(report["k_difference_pcm"], 0.0, places=8)
        self.assertAlmostEqual(
            report["maximum_region_group_flux_error"],
            0.0,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
