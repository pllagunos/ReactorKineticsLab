from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from reactor_backend.multigroup_diffusion import ConcentricMeshSpacing
from reactor_backend.multigroup_diffusion_cache import (
    DiffusionCacheSettings,
    prepare_concentric_diffusion_cache,
)
from reactor_backend.openmc_mgxs_adapter import (
    load_concentric_diffusion_input,
)

try:
    from .test_openmc_mgxs_adapter import AdapterFixture, _export
except ImportError:
    from test_openmc_mgxs_adapter import AdapterFixture, _export


class MultiGroupDiffusionCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = AdapterFixture(copy.deepcopy(_export()))
        self.addCleanup(self.fixture.close)
        self.cache_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache_directory.cleanup)
        self.diffusion_input = load_concentric_diffusion_input(
            self.fixture.path
        )
        self.settings = DiffusionCacheSettings(
            spacing=ConcentricMeshSpacing(
                fuel_radial_cm=1.0,
                core_coolant_radial_cm=50.0,
                moderator_radial_cm=100.0,
                reflector_radial_cm=250.0,
                axial_cm=250.0,
            ),
            max_iter=300,
            tol=1.0e-6,
            max_inner_iter=100,
            inner_tol=1.0e-10,
        )

    def test_cache_persists_group_matrices_and_clean_solution(self):
        first = prepare_concentric_diffusion_cache(
            self.diffusion_input,
            settings=self.settings,
            cache_root=self.cache_directory.name,
        )
        second = prepare_concentric_diffusion_cache(
            self.diffusion_input,
            settings=self.settings,
            cache_root=self.cache_directory.name,
        )

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertAlmostEqual(
            first.clean_solution["k_eff"],
            second.clean_solution["k_eff"],
            places=12,
        )
        self.assertEqual(
            len(second.system.operators),
            self.diffusion_input.group_count,
        )
        self.assertTrue((second.cache_dir / "manifest.json").is_file())
        for group in range(self.diffusion_input.group_count):
            matrix_path = (
                second.cache_dir
                / "operators"
                / f"group_{group + 1:03d}.npz"
            )
            self.assertTrue(matrix_path.is_file())

    def test_mesh_settings_change_cache_fingerprint(self):
        first = prepare_concentric_diffusion_cache(
            self.diffusion_input,
            settings=self.settings,
            cache_root=self.cache_directory.name,
        )
        changed = DiffusionCacheSettings(
            spacing=ConcentricMeshSpacing(
                fuel_radial_cm=0.5,
                core_coolant_radial_cm=50.0,
                moderator_radial_cm=100.0,
                reflector_radial_cm=250.0,
                axial_cm=250.0,
            ),
            max_iter=300,
            tol=1.0e-6,
            max_inner_iter=100,
            inner_tol=1.0e-10,
        )

        second = prepare_concentric_diffusion_cache(
            self.diffusion_input,
            settings=changed,
            cache_root=Path(self.cache_directory.name),
        )

        self.assertNotEqual(first.fingerprint, second.fingerprint)


if __name__ == "__main__":
    unittest.main()
