from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import openmc

from concentric_reactor_reactivity import (
    D2O_MATERIAL_NAME,
    FUEL_MATERIAL_NAME,
    AveragedCaseResult,
    CaseDefinition,
    CaseResult,
    CoefficientDefinition,
    apply_case_to_model,
    average_case_results,
    build_case_definitions,
    build_coefficient_definitions,
    collect_material_baselines,
    compute_coefficient_results,
    material_by_name,
    rho_pcm,
    rho_std_pcm,
    write_case_results_csv,
    write_coefficient_results_csv,
)


def _test_material(name: str, density: float, temperature: float) -> openmc.Material:
    material = openmc.Material(name=name)
    material.add_nuclide("H1", 1.0)
    material.set_density("g/cm3", density)
    material.temperature = temperature
    return material


def _test_model() -> openmc.Model:
    fuel = _test_material(FUEL_MATERIAL_NAME, 12.2, 600.0)
    moderator = _test_material(D2O_MATERIAL_NAME, 1.105, 300.0)
    cell = openmc.Cell(name="test_cell", fill=fuel)
    return openmc.Model(
        geometry=openmc.Geometry([cell]),
        materials=openmc.Materials([fuel, moderator]),
    )


class ReactivityCoefficientScriptTests(unittest.TestCase):
    def test_reactivity_conversion_and_uncertainty(self):
        self.assertAlmostEqual(rho_pcm(1.002), 199.60079840319487)
        self.assertAlmostEqual(rho_std_pcm(1.002, 0.0002), 19.920239361277445)

    def test_collects_material_baselines_and_builds_canonical_cases(self):
        model = _test_model()
        baselines = collect_material_baselines(model)
        definitions = build_coefficient_definitions(
            baselines,
            fuel_temperature_delta_k=50.0,
            moderator_temperature_delta_k=10.0,
            moderator_density_delta_fraction=0.01,
        )
        cases = build_case_definitions(definitions)

        self.assertEqual(baselines[FUEL_MATERIAL_NAME].temperature_k, 600.0)
        self.assertEqual(baselines[D2O_MATERIAL_NAME].density_g_per_cm3, 1.105)
        self.assertEqual([definition.name for definition in definitions], [
            "fuel_temperature",
            "d2o_temperature",
            "d2o_density",
        ])
        self.assertEqual([case.case_id for case in cases], [
            "base",
            "fuel_temperature_minus",
            "fuel_temperature_plus",
            "d2o_temperature_minus",
            "d2o_temperature_plus",
            "d2o_density_minus",
            "d2o_density_plus",
        ])
        self.assertAlmostEqual(cases[-1].perturbed_value, 1.11605)

    def test_apply_case_to_model_changes_only_target_copy(self):
        model = _test_model()
        case = CaseDefinition(
            case_id="d2o_density_plus",
            coefficient_name="d2o_density",
            case_name="plus",
            material_name=D2O_MATERIAL_NAME,
            kind="density",
            baseline_value=1.105,
            perturbed_value=1.11605,
            variable_unit="g/cm3",
        )

        apply_case_to_model(model, case)

        moderator = material_by_name(model, D2O_MATERIAL_NAME)
        fuel = material_by_name(model, FUEL_MATERIAL_NAME)
        self.assertAlmostEqual(moderator.density, 1.11605)
        self.assertAlmostEqual(fuel.density, 12.2)

    def test_missing_required_material_fails_loudly(self):
        model = openmc.Model(materials=openmc.Materials([_test_material("other", 1.0, 300.0)]))

        with self.assertRaisesRegex(ValueError, "Required material"):
            collect_material_baselines(model)

    def test_averages_replicates_with_measurement_and_sample_variance(self):
        results = [
            CaseResult(
                case_id="base",
                coefficient_name="base",
                case_name="base",
                replicate=1,
                seed=1,
                material_name=None,
                kind=None,
                baseline_value=None,
                perturbed_value=None,
                variable_unit=None,
                k_eff=1.001,
                k_eff_std=0.0001,
                rho_pcm=100.0,
                rho_std_pcm=10.0,
                run_dir="/tmp/a",
                statepoint_path="/tmp/a/statepoint.h5",
            ),
            CaseResult(
                case_id="base",
                coefficient_name="base",
                case_name="base",
                replicate=2,
                seed=2,
                material_name=None,
                kind=None,
                baseline_value=None,
                perturbed_value=None,
                variable_unit=None,
                k_eff=1.003,
                k_eff_std=0.0001,
                rho_pcm=120.0,
                rho_std_pcm=10.0,
                run_dir="/tmp/b",
                statepoint_path="/tmp/b/statepoint.h5",
            ),
        ]

        averaged = average_case_results(results)["base"]

        self.assertAlmostEqual(averaged.k_eff, 1.002)
        self.assertAlmostEqual(averaged.rho_pcm, 110.0)
        self.assertGreater(averaged.rho_std_pcm, 7.0)

    def test_computes_temperature_and_density_coefficients(self):
        base = AveragedCaseResult(
            case_id="base",
            coefficient_name="base",
            case_name="base",
            material_name=None,
            kind=None,
            baseline_value=None,
            perturbed_value=None,
            variable_unit=None,
            replicate_count=1,
            k_eff=1.001,
            k_eff_std=0.0001,
            rho_pcm=100.0,
            rho_std_pcm=10.0,
        )
        minus = AveragedCaseResult(
            case_id="d2o_density_minus",
            coefficient_name="d2o_density",
            case_name="minus",
            material_name=D2O_MATERIAL_NAME,
            kind="density",
            baseline_value=1.105,
            perturbed_value=1.09395,
            variable_unit="g/cm3",
            replicate_count=1,
            k_eff=0.999,
            k_eff_std=0.0001,
            rho_pcm=80.0,
            rho_std_pcm=3.0,
        )
        plus = AveragedCaseResult(
            case_id="d2o_density_plus",
            coefficient_name="d2o_density",
            case_name="plus",
            material_name=D2O_MATERIAL_NAME,
            kind="density",
            baseline_value=1.105,
            perturbed_value=1.11605,
            variable_unit="g/cm3",
            replicate_count=1,
            k_eff=1.002,
            k_eff_std=0.0001,
            rho_pcm=120.0,
            rho_std_pcm=4.0,
        )
        definition = CoefficientDefinition(
            name="d2o_density",
            label="D2O moderator density",
            material_name=D2O_MATERIAL_NAME,
            kind="density",
            baseline_value=1.105,
            delta_value=0.01105,
            variable_unit="g/cm3",
            coefficient_unit="pcm/(g/cm3)",
        )

        result = compute_coefficient_results(
            [definition],
            {"base": base, "d2o_density_minus": minus, "d2o_density_plus": plus},
        )[0]

        self.assertAlmostEqual(result.coefficient_value, 40.0 / 0.0221)
        self.assertAlmostEqual(result.coefficient_std, 5.0 / 0.0221)
        self.assertAlmostEqual(result.coefficient_per_percent_density, 20.0)
        self.assertAlmostEqual(result.coefficient_per_percent_density_std, 2.5)

    def test_writes_result_csv_files(self):
        case_result = CaseResult(
            case_id="base",
            coefficient_name="base",
            case_name="base",
            replicate=1,
            seed=1,
            material_name=None,
            kind=None,
            baseline_value=None,
            perturbed_value=None,
            variable_unit=None,
            k_eff=1.001,
            k_eff_std=0.0001,
            rho_pcm=99.9001,
            rho_std_pcm=9.98,
            run_dir="/tmp/run",
            statepoint_path="/tmp/run/statepoint.h5",
        )
        coefficient = compute_coefficient_results(
            [
                CoefficientDefinition(
                    name="fuel_temperature",
                    label="Fuel temperature",
                    material_name=FUEL_MATERIAL_NAME,
                    kind="temperature",
                    baseline_value=600.0,
                    delta_value=50.0,
                    variable_unit="K",
                    coefficient_unit="pcm/K",
                )
            ],
            {
                "fuel_temperature_minus": AveragedCaseResult(
                    "fuel_temperature_minus",
                    "fuel_temperature",
                    "minus",
                    FUEL_MATERIAL_NAME,
                    "temperature",
                    600.0,
                    550.0,
                    "K",
                    1,
                    1.0,
                    0.0001,
                    50.0,
                    5.0,
                ),
                "fuel_temperature_plus": AveragedCaseResult(
                    "fuel_temperature_plus",
                    "fuel_temperature",
                    "plus",
                    FUEL_MATERIAL_NAME,
                    "temperature",
                    600.0,
                    650.0,
                    "K",
                    1,
                    1.0,
                    0.0001,
                    150.0,
                    5.0,
                ),
            },
        )[0]

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            cases_path = write_case_results_csv([case_result], output_dir)
            coefficients_path = write_coefficient_results_csv([coefficient], output_dir)

            with cases_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["case_id"], "base")
            self.assertEqual(rows[0]["k_eff"], "1.001")

            with coefficients_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["name"], "fuel_temperature")
            self.assertEqual(rows[0]["coefficient_unit"], "pcm/K")
            self.assertEqual(rows[0]["coefficient_value"], "1")


if __name__ == "__main__":
    unittest.main()
