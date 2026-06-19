from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import openmc

from mgxs_export import (
    MGXSExportConfig,
    _validate_statepoint_domains,
    attach_mgxs_tallies,
    publish_group_sweep,
)


def _model() -> openmc.Model:
    material = openmc.Material()
    material.add_nuclide("H1", 1.0)
    cell = openmc.Cell(name="test_cell", fill=material)
    geometry = openmc.Geometry([cell])
    return openmc.Model(geometry=geometry)


def _bounded_fuel_model() -> openmc.Model:
    material = openmc.Material()
    material.add_nuclide("H1", 1.0)
    cylinder = openmc.ZCylinder(r=10.0)
    lower = openmc.ZPlane(z0=-20.0)
    upper = openmc.ZPlane(z0=20.0)
    cell = openmc.Cell(
        name="fuel_ring_1",
        fill=material,
        region=-cylinder & +lower & -upper,
    )
    return openmc.Model(geometry=openmc.Geometry([cell]))


class MGXSExportContractTests(unittest.TestCase):
    def test_statepoint_domain_check_rejects_mismatched_cell_ids(self):
        class FakeSummary:
            _fast_cells = {1: object()}

        class FakeStatePoint:
            summary = FakeSummary()

        class FakeCell:
            def __init__(self, cell_id: int):
                self.id = cell_id

        with self.assertRaisesRegex(
            ValueError,
            "not a plain continuous-energy reactor statepoint",
        ):
            _validate_statepoint_domains(
                FakeStatePoint(),
                {"core_central_moderator_channel": FakeCell(74)},
                statepoint_path=Path("/tmp/plain/statepoint.100.h5"),
                model_xml_path=Path("/tmp/export/reactor_run/model.xml"),
            )

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

    def test_resolved_bounded_geometry_gets_cylindrical_power_tally(self):
        config = MGXSExportConfig(
            domain_definitions=(("core_fuel_ring_1", "fuel_ring_1"),),
            reference_power_mesh_radial_bins=4,
            reference_power_mesh_axial_bins=5,
        )

        attached = attach_mgxs_tallies(_bounded_fuel_model(), config)
        power_tally = attached[8]
        metadata = attached[-1]

        self.assertIsNotNone(power_tally)
        assert power_tally is not None
        self.assertEqual(power_tally.scores, ["kappa-fission"])
        self.assertEqual(power_tally.name, "reference-kappa-fission-mesh")
        mesh = power_tally.filters[0].mesh
        self.assertIsInstance(mesh, openmc.CylindricalMesh)
        self.assertEqual(mesh.dimension, (4, 1, 5))
        self.assertEqual(
            metadata["auxiliary_tallies"]["power_mesh"]["radial_bins"],
            4,
        )
        self.assertEqual(
            metadata["auxiliary_tallies"]["power_mesh"]["axial_bins"],
            5,
        )

    def test_publishes_portable_group_sweep_with_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "build" / "group_sweep"
            published_root = root / "reference_data" / "group_sweep"
            group_root = raw_root / "group_4"
            source_json = group_root / "outputs" / "mgxs_constants.json"
            source_model = group_root / "reactor_run" / "model.xml"
            validation_json = (
                group_root
                / "mg_mode_validation"
                / "validation_results.json"
            )
            source_json.parent.mkdir(parents=True)
            source_model.parent.mkdir(parents=True)
            validation_json.parent.mkdir(parents=True)
            export = {
                "config": {
                    "energy_group_edges_ev": [0.0, 1.0, 2.0, 3.0, 4.0],
                    "legendre_order": 0,
                    "scatter_correction": None,
                    "validation_scatter_correction": "P0",
                },
                "run": {
                    "keff": {"mean": 1.001, "std_dev": 0.0001},
                    "reactivity_pcm": 99.9,
                },
            }
            source_json.write_text(
                json.dumps(export),
                encoding="utf-8",
            )
            source_model.write_text("<model/>\n", encoding="utf-8")
            validation_json.write_text(
                json.dumps(
                    {
                        "settings": {
                            "group_count": 4,
                            "scatter_correction": "P0",
                        },
                        "run": {
                            "keff": {"mean": 1.0009, "std_dev": 0.0002},
                            "reactivity_pcm": 89.9,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = publish_group_sweep(
                raw_root,
                published_root,
                group_counts=(4,),
            )

            self.assertEqual(result["group_counts"], (4,))
            published_json = (
                published_root
                / "group_4"
                / "outputs"
                / "mgxs_constants.json"
            )
            published_model = (
                published_root / "group_4" / "reactor_run" / "model.xml"
            )
            self.assertEqual(
                published_json.read_bytes(),
                source_json.read_bytes(),
            )
            self.assertEqual(
                published_model.read_bytes(),
                source_model.read_bytes(),
            )
            manifest = json.loads(
                (published_root / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["group_counts"], [4])
            group = manifest["groups"][0]
            self.assertEqual(group["group_count"], 4)
            self.assertEqual(
                group["multigroup_validation"]["keff"]["mean"],
                1.0009,
            )
            for metadata in group["files"].values():
                self.assertEqual(len(metadata["sha256"]), 64)
                self.assertGreater(metadata["bytes"], 0)

    def test_failed_publication_preserves_existing_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "build" / "group_sweep"
            published_root = root / "reference_data" / "group_sweep"
            published_root.mkdir(parents=True)
            sentinel = published_root / "sentinel.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            source_json = (
                raw_root
                / "group_4"
                / "outputs"
                / "mgxs_constants.json"
            )
            source_model = (
                raw_root / "group_4" / "reactor_run" / "model.xml"
            )
            source_json.parent.mkdir(parents=True)
            source_model.parent.mkdir(parents=True)
            source_json.write_text(
                json.dumps(
                    {
                        "config": {
                            "energy_group_edges_ev": [0.0, 1.0, 2.0],
                        }
                    }
                ),
                encoding="utf-8",
            )
            source_model.write_text("<model/>\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "contains a 2-group",
            ):
                publish_group_sweep(
                    raw_root,
                    published_root,
                    group_counts=(4,),
                )

            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "keep\n",
            )


if __name__ == "__main__":
    unittest.main()
