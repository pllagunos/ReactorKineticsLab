from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reactor_backend.reactivity import (
    ThermalFeedbackInput,
    compute_reactivity,
    compute_thermal_feedback,
)
from reactor_backend.reactivity_coefficients import (
    REACTIVITY_COEFFICIENTS,
    load_reactivity_coefficients,
)


class ReactivityFeedbackTests(unittest.TestCase):
    def test_loads_openmc_reactivity_coefficients(self):
        coefficients = REACTIVITY_COEFFICIENTS

        self.assertAlmostEqual(coefficients.fuel_temperature_base_k, 600.0)
        self.assertAlmostEqual(coefficients.fuel_temperature_pcm_per_k, -2.138272335)
        self.assertAlmostEqual(coefficients.moderator_temperature_base_k, 300.0)
        self.assertAlmostEqual(
            coefficients.moderator_temperature_pcm_per_k,
            -10.72767894,
        )
        self.assertAlmostEqual(coefficients.moderator_density_base_g_per_cm3, 1.105)
        self.assertAlmostEqual(
            coefficients.moderator_density_pcm_per_g_per_cm3,
            20993.04779,
        )

    def test_thermal_feedback_is_fmi_only(self):
        feedback = compute_thermal_feedback(
            ThermalFeedbackInput(
                source="fallback",
                fuel_temperature_k=650.0,
                moderator_temperature_k=310.0,
                moderator_density_g_per_cm3=1.11605,
            )
        )

        self.assertFalse(feedback.applied)
        self.assertEqual(feedback.total_pcm, 0.0)

    def test_thermal_feedback_uses_reset_reference(self):
        reference = ThermalFeedbackInput(
            source="fmu",
            fuel_temperature_k=600.0,
            moderator_temperature_k=300.0,
            moderator_density_g_per_cm3=1.105,
        )
        current = ThermalFeedbackInput(
            source="fmu",
            fuel_temperature_k=650.0,
            moderator_temperature_k=310.0,
            moderator_density_g_per_cm3=1.11605,
        )

        feedback = compute_thermal_feedback(current, reference)

        self.assertTrue(feedback.applied)
        self.assertAlmostEqual(feedback.fuel_temperature_pcm, -106.91361675)
        self.assertAlmostEqual(feedback.moderator_temperature_pcm, -107.2767894)
        self.assertAlmostEqual(feedback.moderator_density_pcm, 231.9731780795008)
        self.assertAlmostEqual(feedback.total_pcm, 17.782771929500778)

        zeroed = compute_reactivity(
            0.0,
            False,
            thermal_feedback=reference,
            reference_feedback=reference,
        )
        self.assertTrue(zeroed.thermalFeedbackApplied)
        self.assertAlmostEqual(zeroed.thermalFeedbackPcm, 0.0)

    def test_rejects_missing_coefficient_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reactivity_coefficients.csv"
            path.write_text(
                "name,baseline_value,coefficient_value\n"
                "fuel_temperature,600,-2.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required rows"):
                load_reactivity_coefficients(path)


if __name__ == "__main__":
    unittest.main()
