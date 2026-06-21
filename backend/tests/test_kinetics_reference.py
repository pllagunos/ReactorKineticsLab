from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reactor_backend.kinetics_reference import (
    KINETICS_REFERENCE,
    load_kinetics_reference,
)


class KineticsReferenceTests(unittest.TestCase):
    def test_loads_group4_mgxs_kinetics_reference(self):
        reference = KINETICS_REFERENCE

        self.assertEqual(len(reference.delayed_groups), 6)
        self.assertAlmostEqual(reference.beta_effective, 0.006789108829782777)
        self.assertAlmostEqual(reference.beta_effective_pcm, 678.9108829782777)
        self.assertAlmostEqual(
            reference.neutron_generation_time_seconds,
            0.004172128337116839,
        )
        self.assertAlmostEqual(reference.delayed_groups[0].beta, 0.0002279170941633448)
        self.assertGreater(reference.delayed_groups[0].decay_constant, 0.0133)
        self.assertLess(reference.delayed_groups[0].decay_constant, 0.0134)

    def test_rejects_missing_prompt_generation_time(self):
        payload = {
            "run": {},
            "domains": {
                "core_fuel_ring_1": {
                    "group_constants": {"nu-fission": {"mean": [1.0]}},
                    "genfoam_aux": {"integral_flux": [1.0]},
                    "delayed_neutrons": {
                        "beta_total_by_delayed_group": [0.001],
                        "decay_rate_per_s_by_delayed_group": [0.1],
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mgxs_constants.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "prompt_generation_time_s"):
                load_kinetics_reference(path)


if __name__ == "__main__":
    unittest.main()
