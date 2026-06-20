from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from reactor_backend.multigroup_service import (
    MultigroupDiffusionService,
    _SolvedState,
)
from reactor_backend.power_shape import PowerShapeCorrection


class MultigroupServiceTests(unittest.TestCase):
    def test_cached_rodded_state_preserves_rod_metadata(self):
        service = MultigroupDiffusionService()
        fake_system = SimpleNamespace(x_insert=0.5)
        fake_solution = {"phi_groups": np.asarray([1.0])}
        service._prepared = SimpleNamespace()
        service._rod_cache[0.5] = (fake_system, fake_solution)

        state = service._solve_for_rod(50.0)

        self.assertIs(state.system, fake_system)
        self.assertIs(state.solution, fake_solution)
        self.assertTrue(state.cached)
        self.assertAlmostEqual(state.rod_insertion_percent, 50.0)

    def test_rodded_response_uses_fixed_clean_power_factor(self):
        service = MultigroupDiffusionService()
        mesh = SimpleNamespace(
            nr=1,
            nz=1,
            cell_count=1,
            volumes=np.asarray([1.0]),
            r_edges=np.asarray([0.0, 1.0]),
            z_edges=np.asarray([0.0, 1.0]),
            r_grid=np.asarray([0.5]),
            z_grid=np.asarray([0.5]),
        )
        system = SimpleNamespace(
            group_count=1,
            cell_count=1,
            mesh=mesh,
            x_insert=0.5,
        )
        solution = {
            "phi_groups": np.asarray([[[1.0]]]),
            "k_eff": 0.99,
            "iterations": 4,
            "timings_s": {"total": 0.25},
        }
        diffusion_input = SimpleNamespace(
            ce_reference=SimpleNamespace(power_mesh=object()),
            openmc_reference={"keff": 1.0, "keff_std_dev": 1.0e-5},
            geometry={
                "core_radius_cm": 1.0,
                "moderator_radius_cm": 1.0,
                "reflector_radius_cm": 1.0,
                "core_height_cm": 1.0,
                "outer_height_cm": 1.0,
            },
            energy_group_edges_ev=[1.0, 0.0],
            regions={"fuel": object()},
        )
        service._prepared = SimpleNamespace(
            diffusion_input=diffusion_input,
            manifest={"settings": {"spacing": {"axial_cm": 1.0}}},
            sph_factors=None,
        )
        service._clean_power_shape = PowerShapeCorrection(
            corrected_power_density=np.asarray([[2.0]]),
            corrected_power_rate=np.asarray([[2.0]]),
            correction_factor=np.asarray([[2.0]]),
            reference_power_shape=np.asarray([[1.0]]),
            diffusion_power_shape=np.asarray([[1.0]]),
            active_bins=np.asarray([[True]]),
            reference_total=1.0,
            diffusion_total=1.0,
            corrected_total=2.0,
        )
        service._power_density_for = lambda _system, _solution: np.asarray([[3.0]])  # type: ignore[method-assign]

        response = service._response(
            _SolvedState(
                system=system,
                solution=solution,
                cached=True,
                rod_insertion_percent=50.0,
            )
        )

        self.assertAlmostEqual(response.metadata.rodInsertionPercent, 50.0)
        self.assertTrue(response.metadata.roddedSolveCached)
        self.assertTrue(response.metadata.cleanCorrectionApplied)
        self.assertEqual(response.heatmapPower, [[1.0]])
        self.assertAlmostEqual(
            response.metadata.powerShapeCorrectionDiffusionTotal,
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
