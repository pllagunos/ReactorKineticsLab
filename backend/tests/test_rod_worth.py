from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reactor_backend.reactivity import compute_reactivity
from reactor_backend.rod_worth import ROD_WORTH_TABLE, load_rod_worth_table


class RodWorthTests(unittest.TestCase):
    def test_loads_openmc_csv_reference(self):
        table = ROD_WORTH_TABLE

        self.assertAlmostEqual(table.clean_excess_pcm, 224.084)
        self.assertAlmostEqual(table.full_insertion_worth_pcm, 1350.73)
        self.assertGreater(table.critical_insertion_percent, 26.0)
        self.assertLess(table.critical_insertion_percent, 32.0)

    def test_interpolates_and_clamps_reactivity(self):
        self.assertAlmostEqual(
            ROD_WORTH_TABLE.interpolate_rod_worth_pcm(0.0),
            0.0,
        )
        midpoint = ROD_WORTH_TABLE.interpolate_rod_worth_pcm(0.5)
        self.assertLess(midpoint, -535.0)
        self.assertGreater(midpoint, -663.0)
        self.assertAlmostEqual(
            ROD_WORTH_TABLE.interpolate_rod_worth_pcm(2.0),
            -1350.73,
        )

        snapshot = compute_reactivity(1000.0, scram_latched=False)
        self.assertAlmostEqual(snapshot.rodInsertionPercent, 100.0)
        self.assertAlmostEqual(snapshot.rodContributionPcm, -1350.73)
        self.assertAlmostEqual(
            snapshot.totalPcm,
            ROD_WORTH_TABLE.clean_excess_pcm - 1350.73,
        )

    def test_rejects_malformed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("insertion_fraction,rho_total_pcm\n0,0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "rod-worth CSV"):
                load_rod_worth_table(path)


if __name__ == "__main__":
    unittest.main()
