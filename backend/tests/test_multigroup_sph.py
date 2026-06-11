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
    SphFactorSet,
    SphSettings,
    build_sph_corrected_system,
    corrected_regions,
    fit_sph_factors,
    qualify_mesh,
    region_integrated_flux,
    sph_source_fingerprint,
)
from reactor_backend.openmc_mgxs_adapter import (
    ContinuousEnergyReference,
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
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            fit_sph_factors(
                self._diffusion_reference_input(),
                spacing=self.spacing,
                settings=SphSettings(
                    max_iterations=1,
                    stable_iterations=2,
                ),
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
