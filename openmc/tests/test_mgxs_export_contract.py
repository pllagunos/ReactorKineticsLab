from __future__ import annotations

import unittest

import openmc

from mgxs_export import MGXSExportConfig, attach_mgxs_tallies


def _model() -> openmc.Model:
    material = openmc.Material()
    material.add_nuclide("H1", 1.0)
    cell = openmc.Cell(name="test_cell", fill=material)
    geometry = openmc.Geometry([cell])
    return openmc.Model(geometry=geometry)


class MGXSExportContractTests(unittest.TestCase):
    def test_canonical_and_validation_libraries_use_distinct_corrections(self):
        config = MGXSExportConfig(
            domain_definitions=(("test", "test_cell"),),
            energy_group_edges_ev=(0.0, 0.625, 20.0e6),
            legendre_order=0,
            scatter_correction=None,
            validation_scatter_correction="P0",
        )

        attached = attach_mgxs_tallies(_model(), config)
        canonical, validation = attached[:2]
        metadata = attached[-1]

        self.assertIsNone(canonical.correction)
        self.assertEqual(validation.correction, "P0")
        self.assertIn("scatter matrix", canonical.mgxs_types)
        self.assertNotIn("scatter matrix", validation.mgxs_types)
        self.assertIn("consistent scatter matrix", validation.mgxs_types)
        self.assertEqual(metadata["canonical_scatter_correction"], None)
        self.assertEqual(metadata["validation_scatter_correction"], "P0")

        domain = validation.domains[0]
        corrected = validation.get_mgxs(
            domain, "consistent scatter matrix"
        )
        self.assertEqual(corrected.correction, "P0")
        self.assertIn("correction", corrected.tallies)
        self.assertIn("flux (analog)", corrected.tallies)

    def test_rejects_corrected_canonical_export(self):
        config = MGXSExportConfig(
            domain_definitions=(("test", "test_cell"),),
            scatter_correction="P0",
        )
        with self.assertRaisesRegex(ValueError, "canonical MGXS export"):
            attach_mgxs_tallies(_model(), config)

    def test_rejects_p0_validation_with_higher_legendre_order(self):
        config = MGXSExportConfig(
            domain_definitions=(("test", "test_cell"),),
            legendre_order=1,
            validation_scatter_correction="P0",
        )
        with self.assertRaisesRegex(ValueError, "ignores P0 correction"):
            attach_mgxs_tallies(_model(), config)


if __name__ == "__main__":
    unittest.main()
