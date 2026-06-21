import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave

from .schemas import ThermalSnapshot


AXIAL_NAME_RE = re.compile(r"^axialPowerFractions\[(\d+)\]$")

REQUIRED_FMU_VARIABLES = frozenset(
    {
        "totalPower",
        "T_inlet",
        "T_outlet",
        "T_fuelCenterlineMax",
        "T_fuelEff",
        "T_moderatorEff",
        "rho_m_eff_SI",
        "rho_m_eff",
        "massFlow",
        "dp_core",
    }
)


@dataclass(frozen=True)
class ThermalAdapterConfig:
    model_name: str = "ResearchReactorThermalHydraulics"
    omc_command: str = "omc"
    fmi_flags: str = "s:cvode"

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def modelica_dir(self) -> Path:
        return self.repo_root / "modelica"

    @property
    def model_file(self) -> Path:
        return self.modelica_dir / f"{self.model_name}.mo"

    @property
    def build_dir(self) -> Path:
        return self.modelica_dir / "build"

    @property
    def fmu_path(self) -> Path:
        return self.build_dir / f"{self.model_name}.fmu"


class ThermalAdapter:
    _FALLBACK_INLET_BASE_K = 298.15
    _FALLBACK_INLET_GAIN_K_PER_MW = (300.675 - 298.15) / 20.0
    _FALLBACK_DELTA_T_K_PER_MW = (320.854 - 300.675) / 20.0
    _FALLBACK_TIME_CONSTANT_SECONDS = 25.0
    _FALLBACK_MASS_FLOW_KG_S = 237.0
    _FALLBACK_CORE_DP_PA = 134.43
    _FALLBACK_FUEL_TO_MODERATOR_DELTA_K_AT_20_MW = 52.7
    _FALLBACK_D2O_DENSITY_REF_KG_M3 = 1105.0
    _FALLBACK_D2O_T_REF_K = 300.0
    _FALLBACK_D2O_P_REF_PA = 4.0e5
    _FALLBACK_D2O_BETA_1_K = 3.0e-4
    _FALLBACK_D2O_KAPPA_1_PA = 4.5e-10

    def __init__(self, config: Optional[ThermalAdapterConfig] = None) -> None:
        self._config = config or ThermalAdapterConfig()
        self._fmu: Optional[FMU2Slave] = None
        self._unzip_dir: Optional[str] = None
        self._time_seconds = 0.0
        self._vr_total_power: Optional[int] = None
        self._vr_inlet_temperature: Optional[int] = None
        self._vr_outlet_temperature: Optional[int] = None
        self._vr_fuel_maximum_temperature: Optional[int] = None
        self._vr_fuel_temperature: Optional[int] = None
        self._vr_moderator_temperature: Optional[int] = None
        self._vr_moderator_density_si: Optional[int] = None
        self._vr_moderator_density_cgs: Optional[int] = None
        self._vr_mass_flow: Optional[int] = None
        self._vr_core_dp: Optional[int] = None
        self._vr_axial_profile: list[int] = []
        self._axial_profile_values: list[float] = []
        self._fallback_active = False
        self._fallback_inlet_temperature_k = self._FALLBACK_INLET_BASE_K
        self._fallback_outlet_temperature_k = self._FALLBACK_INLET_BASE_K
        self._latest = self._unavailable_snapshot(
            power_mw=0.0,
            message="Thermal adapter not initialized.",
        )

    def reset(self, initial_power_mw: Optional[float] = None) -> ThermalSnapshot:
        self.close()
        self._time_seconds = 0.0
        self._fallback_active = False
        self._latest = self._unavailable_snapshot(
            power_mw=initial_power_mw or 0.0,
            message="Thermal adapter not initialized.",
        )
        if initial_power_mw is not None:
            self._ensure_instance(initial_power_mw)
            if not self._latest.available:
                self._activate_fallback(
                    power_mw=initial_power_mw,
                    message=self._latest.message or "Using fallback thermal model.",
                )
        return self._latest

    def step(
        self,
        power_mw: float,
        dt_seconds: float,
        reported_power_mw: Optional[float] = None,
    ) -> ThermalSnapshot:
        snapshot_power_mw = power_mw if reported_power_mw is None else reported_power_mw
        if dt_seconds < 0:
            return self._unavailable_snapshot(
                power_mw=snapshot_power_mw,
                message="Thermal adapter received a negative time step.",
            )

        if self._fallback_active:
            self._advance_fallback(
                power_mw,
                dt_seconds,
                reported_power_mw=snapshot_power_mw,
            )
            return self._latest

        if not self._ensure_instance(power_mw):
            self._activate_fallback(
                power_mw=power_mw,
                message=self._latest.message or "Using fallback thermal model.",
                reported_power_mw=snapshot_power_mw,
            )
            return self._latest

        try:
            self._set_inputs(power_mw)
            if dt_seconds > 0:
                assert self._fmu is not None
                self._fmu.doStep(
                    currentCommunicationPoint=self._time_seconds,
                    communicationStepSize=dt_seconds,
                )
                self._time_seconds += dt_seconds
            self._latest = self._read_outputs(snapshot_power_mw)
        except Exception as exc:
            prior_inlet = self._latest.inletTemperatureK
            prior_outlet = self._latest.outletTemperatureK
            self.close()
            self._activate_fallback(
                power_mw=power_mw,
                message=f"Thermal FMU step failed; using fallback thermal model: {exc}",
                inlet_temperature_k=prior_inlet,
                outlet_temperature_k=prior_outlet,
                reported_power_mw=snapshot_power_mw,
            )
            self._advance_fallback(
                power_mw,
                dt_seconds,
                reported_power_mw=snapshot_power_mw,
            )

        return self._latest

    def get_snapshot(self) -> ThermalSnapshot:
        return self._latest

    def set_axial_fractions(self, fractions: list[float]) -> None:
        """Update the 8 axial power fractions forwarded to the FMU on the next step."""
        if len(fractions) == 8:
            self._axial_profile_values = list(fractions)

    def close(self) -> None:
        if self._fmu is not None:
            try:
                self._fmu.terminate()
            except Exception:
                pass
            try:
                self._fmu.freeInstance()
            except Exception:
                pass
        if self._unzip_dir:
            shutil.rmtree(self._unzip_dir, ignore_errors=True)

        self._fmu = None
        self._unzip_dir = None
        self._vr_total_power = None
        self._vr_inlet_temperature = None
        self._vr_outlet_temperature = None
        self._vr_fuel_maximum_temperature = None
        self._vr_fuel_temperature = None
        self._vr_moderator_temperature = None
        self._vr_moderator_density_si = None
        self._vr_moderator_density_cgs = None
        self._vr_mass_flow = None
        self._vr_core_dp = None
        self._vr_axial_profile = []
        self._axial_profile_values = []

    def _ensure_instance(self, initial_power_mw: float) -> bool:
        if self._fmu is not None:
            return True

        try:
            fmu_path = self._ensure_fmu()
            model_description = read_model_description(str(fmu_path))
            if model_description.coSimulation is None:
                raise RuntimeError(
                    f"{self._config.model_name} FMU does not expose a co-simulation interface."
                )

            variable_refs: dict[str, int] = {}
            axial_refs: list[tuple[int, int]] = []
            for variable in model_description.modelVariables:
                variable_refs[variable.name] = variable.valueReference
                match = AXIAL_NAME_RE.match(variable.name)
                if match:
                    axial_refs.append((int(match.group(1)), variable.valueReference))

            axial_refs.sort(key=lambda item: item[0])
            if not axial_refs:
                raise RuntimeError("No axialPowerFractions input variables were found in the TH FMU.")

            self._vr_total_power = self._required_variable_ref(variable_refs, "totalPower")
            self._vr_inlet_temperature = self._required_variable_ref(variable_refs, "T_inlet")
            self._vr_outlet_temperature = self._required_variable_ref(variable_refs, "T_outlet")
            self._vr_fuel_maximum_temperature = self._required_variable_ref(variable_refs, "T_fuelCenterlineMax")
            self._vr_fuel_temperature = self._required_variable_ref(variable_refs, "T_fuelEff")
            self._vr_moderator_temperature = self._required_variable_ref(variable_refs, "T_moderatorEff")
            self._vr_moderator_density_si = self._required_variable_ref(variable_refs, "rho_m_eff_SI")
            self._vr_moderator_density_cgs = self._required_variable_ref(variable_refs, "rho_m_eff")
            self._vr_mass_flow = self._required_variable_ref(variable_refs, "massFlow")
            self._vr_core_dp = self._required_variable_ref(variable_refs, "dp_core")
            self._vr_axial_profile = [value_ref for _, value_ref in axial_refs]
            self._axial_profile_values = [1.0 / len(self._vr_axial_profile)] * len(self._vr_axial_profile)

            self._unzip_dir = extract(str(fmu_path))
            fmu = FMU2Slave(
                guid=model_description.guid,
                unzipDirectory=self._unzip_dir,
                modelIdentifier=model_description.coSimulation.modelIdentifier,
                instanceName="reactor_th_adapter",
            )
            self._fmu = fmu
            fmu.instantiate()
            fmu.setupExperiment(startTime=0.0)
            fmu.enterInitializationMode()
            self._set_inputs(initial_power_mw)
            fmu.exitInitializationMode()
            self._latest = self._read_outputs(initial_power_mw)
            return True
        except Exception as exc:
            self.close()
            self._latest = self._unavailable_snapshot(
                power_mw=initial_power_mw,
                message=f"Thermal FMU initialization failed: {exc}",
            )
            return False

    def _activate_fallback(
        self,
        power_mw: float,
        message: str,
        inlet_temperature_k: Optional[float] = None,
        outlet_temperature_k: Optional[float] = None,
        reported_power_mw: Optional[float] = None,
    ) -> None:
        snapshot_power_mw = power_mw if reported_power_mw is None else reported_power_mw
        self._fallback_active = True
        target_inlet, target_outlet = self._fallback_targets(power_mw)
        self._fallback_inlet_temperature_k = (
            inlet_temperature_k if inlet_temperature_k is not None else target_inlet
        )
        self._fallback_outlet_temperature_k = (
            outlet_temperature_k if outlet_temperature_k is not None else target_outlet
        )
        moderator_temperature_k = self._fallback_moderator_temperature()
        moderator_density_kg_m3 = self._fallback_moderator_density_kg_m3(moderator_temperature_k)
        fuel_temperature_k = self._fallback_fuel_temperature(power_mw, moderator_temperature_k)
        self._latest = ThermalSnapshot(
            available=True,
            source="fallback",
            timeSeconds=self._time_seconds,
            powerMw=snapshot_power_mw,
            inletTemperatureK=self._fallback_inlet_temperature_k,
            outletTemperatureK=self._fallback_outlet_temperature_k,
            fuelMaximumTemperatureK=fuel_temperature_k,
            fuelTemperatureK=fuel_temperature_k,
            moderatorTemperatureK=moderator_temperature_k,
            moderatorDensityKgPerM3=moderator_density_kg_m3,
            moderatorDensityGPerCm3=moderator_density_kg_m3 / 1000.0,
            massFlowKgPerSecond=self._FALLBACK_MASS_FLOW_KG_S,
            corePressureDropPa=self._FALLBACK_CORE_DP_PA,
            message=message,
            axialPowerFractions=list(self._axial_profile_values),
        )

    def _advance_fallback(
        self,
        power_mw: float,
        dt_seconds: float,
        reported_power_mw: Optional[float] = None,
    ) -> None:
        snapshot_power_mw = power_mw if reported_power_mw is None else reported_power_mw
        target_inlet, target_outlet = self._fallback_targets(power_mw)
        if dt_seconds > 0:
            alpha = 1.0 - 2.718281828459045 ** (
                -dt_seconds / self._FALLBACK_TIME_CONSTANT_SECONDS
            )
            self._fallback_inlet_temperature_k += alpha * (
                target_inlet - self._fallback_inlet_temperature_k
            )
            self._fallback_outlet_temperature_k += alpha * (
                target_outlet - self._fallback_outlet_temperature_k
            )
            self._time_seconds += dt_seconds

        moderator_temperature_k = self._fallback_moderator_temperature()
        moderator_density_kg_m3 = self._fallback_moderator_density_kg_m3(moderator_temperature_k)
        fuel_temperature_k = self._fallback_fuel_temperature(power_mw, moderator_temperature_k)
        self._latest = ThermalSnapshot(
            available=True,
            source="fallback",
            timeSeconds=self._time_seconds,
            powerMw=snapshot_power_mw,
            inletTemperatureK=self._fallback_inlet_temperature_k,
            outletTemperatureK=self._fallback_outlet_temperature_k,
            fuelMaximumTemperatureK=fuel_temperature_k,
            fuelTemperatureK=fuel_temperature_k,
            moderatorTemperatureK=moderator_temperature_k,
            moderatorDensityKgPerM3=moderator_density_kg_m3,
            moderatorDensityGPerCm3=moderator_density_kg_m3 / 1000.0,
            massFlowKgPerSecond=self._FALLBACK_MASS_FLOW_KG_S,
            corePressureDropPa=self._FALLBACK_CORE_DP_PA,
            message=self._latest.message,
            axialPowerFractions=list(self._axial_profile_values),
        )

    def _fallback_targets(self, power_mw: float) -> tuple[float, float]:
        inlet = self._FALLBACK_INLET_BASE_K + self._FALLBACK_INLET_GAIN_K_PER_MW * power_mw
        outlet = inlet + self._FALLBACK_DELTA_T_K_PER_MW * power_mw
        return inlet, outlet

    def _fallback_moderator_temperature(self) -> float:
        return 0.5 * (self._fallback_inlet_temperature_k + self._fallback_outlet_temperature_k)

    def _fallback_fuel_temperature(self, power_mw: float, moderator_temperature_k: float) -> float:
        power_scale = max(power_mw, 0.0) / 20.0
        return moderator_temperature_k + self._FALLBACK_FUEL_TO_MODERATOR_DELTA_K_AT_20_MW * power_scale

    def _fallback_moderator_density_kg_m3(self, moderator_temperature_k: float) -> float:
        density = self._FALLBACK_D2O_DENSITY_REF_KG_M3 * (
            1.0
            - self._FALLBACK_D2O_BETA_1_K * (moderator_temperature_k - self._FALLBACK_D2O_T_REF_K)
            + self._FALLBACK_D2O_KAPPA_1_PA * (4.0e5 - self._FALLBACK_D2O_P_REF_PA)
        )
        return max(density, 0.0)

    def _ensure_fmu(self) -> Path:
        model_file = self._config.model_file
        fmu_path = self._config.fmu_path
        if (
            fmu_path.exists()
            and fmu_path.stat().st_mtime >= model_file.stat().st_mtime
            and self._fmu_is_compatible(fmu_path)
        ):
            return fmu_path

        self._config.build_dir.mkdir(parents=True, exist_ok=True)
        fmu_path.unlink(missing_ok=True)
        export_started_at = model_file.stat().st_mtime
        export_script = self._build_export_script()
        with tempfile.NamedTemporaryFile("w", suffix=".mos", delete=False) as handle:
            handle.write(export_script)
            script_path = Path(handle.name)

        try:
            result = subprocess.run(
                [self._config.omc_command, str(script_path)],
                cwd=self._config.modelica_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            script_path.unlink(missing_ok=True)

        if result.returncode != 0 or not fmu_path.exists():
            output = (result.stdout + "\n" + result.stderr).strip()
            raise RuntimeError(
                "FMU export failed. "
                + (output[-2000:] if output else "No compiler output was captured.")
            )
        if fmu_path.stat().st_mtime < export_started_at:
            raise RuntimeError("FMU export did not produce a fresh FMU artifact.")

        return fmu_path

    def _build_export_script(self) -> str:
        model_name = self._config.model_name
        fmi_flags = self._config.fmi_flags.replace('"', '\\"')
        return (
            f'setCommandLineOptions("--fmiFlags={fmi_flags}");\n'
            'system("mkdir -p build");\n'
            'cd("build");\n'
            'loadModel(Modelica);\n'
            'getErrorString();\n'
            f'loadFile("../{model_name}.mo");\n'
            'getErrorString();\n'
            f'buildModelFMU({model_name}, version="2.0", fmuType="cs");\n'
            'getErrorString();\n'
        )

    def _fmu_is_compatible(self, fmu_path: Path) -> bool:
        if not self._fmu_has_expected_flags(fmu_path):
            return False

        try:
            model_description = read_model_description(str(fmu_path))
        except Exception:
            return False

        if model_description.coSimulation is None:
            return False

        variable_names = {variable.name for variable in model_description.modelVariables}
        if not REQUIRED_FMU_VARIABLES.issubset(variable_names):
            return False

        return any(AXIAL_NAME_RE.match(name) for name in variable_names)

    def _fmu_has_expected_flags(self, fmu_path: Path) -> bool:
        expected = self._config.fmi_flags
        try:
            with zipfile.ZipFile(fmu_path) as archive:
                resource_name = f"resources/{self._config.model_name}_flags.json"
                if resource_name not in archive.namelist():
                    return False
                content = archive.read(resource_name).decode("utf-8", errors="ignore")
        except OSError:
            return False

        return expected in content

    def _set_inputs(self, power_mw: float) -> None:
        assert self._fmu is not None
        assert self._vr_total_power is not None
        self._fmu.setReal([self._vr_total_power], [power_mw * 1e6])
        if self._vr_axial_profile:
            self._fmu.setReal(self._vr_axial_profile, self._axial_profile_values)

    def _read_outputs(self, power_mw: float) -> ThermalSnapshot:
        assert self._fmu is not None
        assert self._vr_inlet_temperature is not None
        assert self._vr_outlet_temperature is not None
        assert self._vr_fuel_maximum_temperature is not None
        assert self._vr_fuel_temperature is not None
        assert self._vr_moderator_temperature is not None
        assert self._vr_moderator_density_si is not None
        assert self._vr_moderator_density_cgs is not None
        assert self._vr_mass_flow is not None
        assert self._vr_core_dp is not None

        values = self._fmu.getReal(
            [
                self._vr_inlet_temperature,
                self._vr_outlet_temperature,
                self._vr_fuel_maximum_temperature,
                self._vr_fuel_temperature,
                self._vr_moderator_temperature,
                self._vr_moderator_density_si,
                self._vr_moderator_density_cgs,
                self._vr_mass_flow,
                self._vr_core_dp,
            ]
        )
        return ThermalSnapshot(
            available=True,
            source="fmu",
            timeSeconds=self._time_seconds,
            powerMw=power_mw,
            inletTemperatureK=values[0],
            outletTemperatureK=values[1],
            fuelMaximumTemperatureK=values[2],
            fuelTemperatureK=values[3],
            moderatorTemperatureK=values[4],
            moderatorDensityKgPerM3=values[5],
            moderatorDensityGPerCm3=values[6],
            massFlowKgPerSecond=values[7],
            corePressureDropPa=values[8],
            message=None,
            axialPowerFractions=list(self._axial_profile_values),
        )

    def _required_variable_ref(self, refs: dict[str, int], name: str) -> int:
        if name not in refs:
            raise RuntimeError(f"Required FMU variable '{name}' was not found.")
        return refs[name]

    def _unavailable_snapshot(self, power_mw: float, message: str) -> ThermalSnapshot:
        return ThermalSnapshot(
            available=False,
            source="unavailable",
            timeSeconds=self._time_seconds,
            powerMw=power_mw,
            message=message,
            axialPowerFractions=list(self._axial_profile_values),
        )
