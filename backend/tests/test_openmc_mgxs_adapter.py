from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reactor_backend.openmc_mgxs_adapter import (
    load_concentric_diffusion_input,
)


DOMAIN_DEFINITIONS = [
    ["core_central_moderator_channel", "central_moderator_channel"],
    ["core_control_rod", "control_rod"],
    ["core_fuel_ring_1", "fuel_ring_1"],
    [
        "core_heavy_water_coolant_and_moderator",
        "heavy_water_coolant_and_moderator",
    ],
    ["moderator", "d2o_tank"],
    ["reflector", "h2o_tank"],
]


MODEL_XML = """\
<model>
  <geometry>
    <surface id="1" type="z-cylinder" coeffs="0 0 50"/>
    <surface id="2" type="z-plane" coeffs="-200"/>
    <surface id="3" type="z-plane" coeffs="200"/>
    <surface id="4" type="z-cylinder" coeffs="0 0 250"/>
    <surface id="5" type="z-plane" coeffs="-300"/>
    <surface id="6" type="z-plane" coeffs="300"/>
    <surface id="7" type="z-cylinder" coeffs="0 0 500"/>
    <surface id="8" type="z-plane" coeffs="-500"/>
    <surface id="9" type="z-plane" coeffs="500"/>
    <surface id="10" type="z-cylinder" coeffs="0 0 4"/>
    <surface id="11" type="z-plane" coeffs="150"/>
    <surface id="12" type="z-cylinder" coeffs="0 0 4.5"/>
    <surface id="13" type="z-cylinder" coeffs="0 0 5"/>
    <surface id="14" type="z-plane" coeffs="-150"/>
    <cell id="1" name="fuel_element" region="-1 2 -3"/>
    <cell id="2" name="d2o_tank" region="-4 5 -6"/>
    <cell id="3" name="h2o_tank" region="-7 8 -9"/>
    <cell id="4" name="control_rod" region="-10 11 -3"/>
    <cell id="5" name="central_moderator_channel" region="-10 2 -11"/>
    <cell id="6" name="heavy_water_coolant_and_moderator" region="-1 2 -3"/>
    <cell id="7" name="fuel_ring_1" region="12 -13 14 -11"/>
  </geometry>
</model>
"""


def _group_constants(
    *,
    scatter: object | None = None,
    consistent_scatter: object | None = None,
    diffusion: list[float] | None = None,
) -> dict:
    transport = [0.5, 1.0]
    constants = {
        "transport": {"mean": transport, "std_dev": [0.0, 0.0]},
        "diffusion-coefficient": {
            "mean": diffusion or [1.0 / (3.0 * value) for value in transport],
            "std_dev": [0.0, 0.0],
        },
        "absorption": {"mean": [0.01, 0.02], "std_dev": [0.0, 0.0]},
        "nu-fission": {"mean": [0.02, 0.03], "std_dev": [0.0, 0.0]},
        "kappa-fission": {"mean": [1.0, 2.0], "std_dev": [0.0, 0.0]},
        "chi": {"mean": [1.0, 0.0], "std_dev": [0.0, 0.0]},
        "scatter matrix": {
            "mean": scatter or [[0.10, 0.04], [0.01, 0.20]],
            "std_dev": [[0.0, 0.0], [0.0, 0.0]],
        },
    }
    if consistent_scatter is not None:
        constants["consistent scatter matrix"] = {
            "mean": consistent_scatter,
            "std_dev": [[0.0, 0.0], [0.0, 0.0]],
        }
    return constants


def _export() -> dict:
    domains = {}
    for label, cell_name in DOMAIN_DEFINITIONS:
        domains[label] = {
            "domain": {"type": "cell", "name": cell_name, "id": len(domains) + 1},
            "group_constants": _group_constants(),
        }
    return {
        "config": {
            "domain_definitions": copy.deepcopy(DOMAIN_DEFINITIONS),
            "energy_group_edges_ev": [0.0, 0.625, 20.0e6],
            "legendre_order": 0,
            "scatter_correction": None,
        },
        "domains": domains,
        "reference": {
            "energy_order": "fast-to-thermal",
            "normalization": "raw OpenMC tally mean per source particle",
            "region_flux": {
                label: {
                    "mean": [1.0, 2.0],
                    "std_dev": [0.01, 0.02],
                }
                for label in domains
            },
            "master_flux": {
                "mean": [float(len(domains)), float(2 * len(domains))],
                "std_dev": [0.01, 0.02],
            },
            "power_mesh": {
                "r_edges_cm": [0.0, 2.5, 5.0],
                "z_edges_cm": [-150.0, 0.0, 150.0],
                "mean": [[1.0, 2.0], [3.0, 4.0]],
                "std_dev": [[0.1, 0.1], [0.1, 0.1]],
            },
        },
        "run": {
            "keff": {"mean": 1.001, "std_dev": 0.0001},
            "reactivity_pcm": 99.9,
        },
    }


class AdapterFixture:
    def __init__(self, export: dict):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_directory.name) / "group_2"
        (self.path / "reactor_run").mkdir(parents=True)
        (self.path / "outputs").mkdir()
        (self.path / "reactor_run" / "model.xml").write_text(
            MODEL_XML, encoding="utf-8"
        )
        (self.path / "outputs" / "mgxs_constants.json").write_text(
            json.dumps(export), encoding="utf-8"
        )

    def close(self) -> None:
        self._temporary_directory.cleanup()


class OpenMCMGXSAdapterTests(unittest.TestCase):
    def load(self, export: dict):
        fixture = AdapterFixture(export)
        self.addCleanup(fixture.close)
        return load_concentric_diffusion_input(fixture.path)

    def test_prefers_consistent_non_nu_scatter_matrix(self):
        export = _export()
        preferred = [[0.3, 0.2], [0.1, 0.4]]
        export["domains"]["core_fuel_ring_1"]["group_constants"] = _group_constants(
            scatter=[[9.0, 9.0], [9.0, 9.0]],
            consistent_scatter=preferred,
        )

        loaded = self.load(export)

        np.testing.assert_allclose(
            loaded.regions["core_fuel_ring_1"].scatter, preferred
        )
        self.assertEqual(
            loaded.validation["domains"]["core_fuel_ring_1"][
                "scatter_matrix_source"
            ],
            "consistent scatter matrix",
        )

    def test_rejects_nu_scatter_as_diffusion_scattering_input(self):
        export = _export()
        for domain in export["domains"].values():
            constants = domain["group_constants"]
            constants["nu-scatter matrix"] = constants.pop("scatter matrix")

        with self.assertRaisesRegex(ValueError, "non-nu scattering matrix"):
            self.load(export)

    def test_collapses_higher_order_scatter_to_p0(self):
        export = _export()
        export["config"]["legendre_order"] = 1
        p0 = [[0.1, 0.2], [0.3, 0.4]]
        p1 = [[1.1, 1.2], [1.3, 1.4]]
        export["domains"]["core_fuel_ring_1"]["group_constants"] = _group_constants(
            scatter=[p0, p1]
        )

        loaded = self.load(export)

        np.testing.assert_allclose(
            loaded.regions["core_fuel_ring_1"].scatter, p0
        )
        self.assertTrue(loaded.summary()["discarded_higher_moments"])

    def test_rejects_corrected_or_ambiguous_exports(self):
        corrected = _export()
        corrected["config"]["scatter_correction"] = "P0"
        with self.assertRaisesRegex(ValueError, "scatter_correction = null"):
            self.load(corrected)

        ambiguous = _export()
        del ambiguous["config"]["scatter_correction"]
        with self.assertRaisesRegex(ValueError, "does not declare"):
            self.load(ambiguous)

    def test_rejects_unresolved_export(self):
        export = _export()
        export["domains"].pop("core_fuel_ring_1")
        export["config"]["domain_definitions"] = [
            item for item in DOMAIN_DEFINITIONS if item[0] != "core_fuel_ring_1"
        ]
        with self.assertRaisesRegex(ValueError, "required MGXS domains"):
            self.load(export)

    def test_rejects_xml_domain_name_mismatch(self):
        export = _export()
        export["domains"]["moderator"]["domain"]["name"] = "wrong_cell"
        with self.assertRaisesRegex(ValueError, "but domain_definitions maps"):
            self.load(export)

    def test_rejects_domain_definition_mismatch(self):
        export = _export()
        export["config"]["domain_definitions"].append(
            ["unused_domain", "fuel_element"]
        )
        with self.assertRaisesRegex(ValueError, "must contain the same labels"):
            self.load(export)

    def test_rejects_invalid_group_shape(self):
        export = _export()
        export["domains"]["moderator"]["group_constants"]["absorption"]["mean"] = [
            0.01
        ]
        with self.assertRaisesRegex(ValueError, "must contain 2 values"):
            self.load(export)

    def test_loads_raw_ce_reference_without_groupwise_normalization(self):
        loaded = self.load(_export())

        self.assertIsNotNone(loaded.ce_reference)
        assert loaded.ce_reference is not None
        np.testing.assert_allclose(
            loaded.ce_reference.region_flux["core_fuel_ring_1"].mean,
            [1.0, 2.0],
        )
        self.assertEqual(
            loaded.ce_reference.energy_order,
            "fast-to-thermal",
        )
        self.assertEqual(loaded.ce_reference.power_mesh.mean.shape, (2, 2))

    def test_rejects_inconsistent_diffusion_coefficient(self):
        export = _export()
        export["domains"]["moderator"]["group_constants"] = _group_constants(
            diffusion=[1.0, 1.0]
        )
        with self.assertRaisesRegex(ValueError, "1/\\(3\\*Sigma_tr\\)"):
            self.load(export)

    def test_builds_resolved_model_without_region_aggregation(self):
        loaded = self.load(copy.deepcopy(_export()))

        model = loaded.build_model(delta_absorption_rod=0.25)

        self.assertEqual(loaded.group_count, 2)
        self.assertEqual(len(loaded.fuel_ring_labels), 1)
        self.assertEqual(len(model.zones), 6)
        self.assertEqual(
            loaded.zone_report[0]["cell_name"],
            "h2o_tank",
        )
        self.assertIn("core_fuel_ring_1", loaded.regions)
        self.assertEqual(model.delta_absorption_rod, 0.25)


if __name__ == "__main__":
    unittest.main()
