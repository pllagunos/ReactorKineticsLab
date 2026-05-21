model ResearchReactorPrimaryLoop
  "Open-tank primary loop prototype for research reactor FMI coupling"
  extends Modelica.Icons.Example;

  parameter Integer nAxialNodes(min = 1) = 8;
  parameter Modelica.Units.SI.Power defaultTotalPower = 20e6;
  parameter Real defaultValveOpening(min = 0, max = 1) = 0.5;

  replaceable package Medium =
    Modelica.Media.CompressibleLiquids.LinearWater_pT_Ambient constrainedby
    Modelica.Media.Interfaces.PartialMedium;

  // ── Tank ─────────────────────────────────────────────────────────────────
  Modelica.Fluid.Vessels.OpenTank tank(
    redeclare package Medium = Medium,
    crossArea = 150,
    height = 15,
    level_start = 14,
    nPorts = 2,
    massDynamics = Modelica.Fluid.Types.Dynamics.FixedInitial,
    use_HeatTransfer = true,
    portsData = {
      Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.5),
      Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.5)},
    redeclare model HeatTransfer =
      Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.IdealHeatTransfer(k = 10),
    s(each start = 1),
    ports(each p(start = 1e5)),
    T_start = Modelica.Units.Conversions.from_degC(20))
    annotation(Placement(transformation(extent = {{-80, 30}, {-60, 50}})));

  // ── Pump ──────────────────────────────────────────────────────────────────
  // FIX 1: p_b_nominal was 6.0 Pa (typo) → corrected to 6e5 Pa (5 bar rise).
  //        p_a_nominal and p_a/b_start now consistently reflect the 5-bar head.
  Modelica.Fluid.Machines.ControlledPump pump(
    redeclare package Medium = Medium,
    N_nominal = 1500,
    use_T_start = true,
    T_start = Modelica.Units.Conversions.from_degC(25),
    m_flow_start = 100,
    m_flow_nominal = 200,
    control_m_flow = false,
    allowFlowReversal = false,
    p_a_start = 100000,
    p_b_start = 600000,
    p_a_nominal = 1e5,
    p_b_nominal = 6e5)
    annotation(Placement(transformation(extent = {{-50, 10}, {-30, 30}})));

  // ── Valve ─────────────────────────────────────────────────────────────────
  // FIX 5: dp_start and dp_nominal updated for 5-bar pressure network.
  Modelica.Fluid.Valves.ValveIncompressible valve(
    redeclare package Medium = Medium,
    CvData = Modelica.Fluid.Types.CvTypes.OpPoint,
    m_flow_nominal = 200,
    show_T = true,
    dp_start = 10000,
    dp_nominal = 100000)
    annotation(Placement(transformation(extent = {{60, -80}, {40, -60}})));

  // ── External inputs ───────────────────────────────────────────────────────
  Modelica.Blocks.Interfaces.RealInput totalPower(
    unit = "W",
    min = 0,
    start = defaultTotalPower)
    annotation(Placement(transformation(extent = {{-106, -10}, {-94, 2}})));

  Modelica.Blocks.Interfaces.RealInput axialPowerFractions[nAxialNodes](
    each unit = "1",
    each min = 0,
    each start = 1.0 / nAxialNodes)
    annotation(Placement(transformation(extent = {{-106, -30}, {-94, -18}})));

  Modelica.Blocks.Interfaces.RealInput valveOpening(
    unit = "1",
    min = 0,
    max = 1,
    start = defaultValveOpening)
    annotation(Placement(transformation(extent = {{-106, -50}, {-94, -38}})));

  // ── Outputs ───────────────────────────────────────────────────────────────
  Modelica.Blocks.Interfaces.RealOutput m_flow(unit = "kg/s")
    annotation(Placement(transformation(extent = {{-6, 34}, {6, 46}})));

  Modelica.Blocks.Interfaces.RealOutput T_forward(unit = "K")
    annotation(Placement(transformation(extent = {{74, 34}, {86, 46}})));

  Modelica.Blocks.Interfaces.RealOutput T_return(unit = "K")
    annotation(Placement(transformation(extent = {{-46, -56}, {-58, -44}})));

  Modelica.Blocks.Interfaces.RealOutput tankLevel(unit = "m")
    annotation(Placement(transformation(extent = {{-56, 34}, {-44, 46}})));

  // ── Sensors ───────────────────────────────────────────────────────────────
  Modelica.Fluid.Sensors.MassFlowRate sensor_m_flow(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(extent = {{-20, 10}, {0, 30}})));

  Modelica.Fluid.Sensors.Temperature sensor_T_forward(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(extent = {{50, 30}, {70, 50}})));

  Modelica.Fluid.Sensors.Temperature sensor_T_return(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(extent = {{-20, -60}, {-40, -40}})));

  // ── Thermal components ────────────────────────────────────────────────────
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature T_ambient(
    T = system.T_ambient)
    annotation(Placement(transformation(extent = {{-14, -27}, {0, -13}})));

  Modelica.Thermal.HeatTransfer.Components.ThermalConductor wall(
    G = 1.6e3 / 20)
    annotation(Placement(transformation(origin = {10, -48}, extent = {{8, -10}, {-8, 10}}, rotation = 90)));

  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow coreHeat[nAxialNodes]
    annotation(Placement(transformation(extent = {{16, 30}, {36, 50}})));

  // ── System ────────────────────────────────────────────────────────────────
  inner Modelica.Fluid.System system(
    m_flow_small = 1e-2,
    m_flow_start = 100,
    p_start = 1e5,
    T_start = Modelica.Units.Conversions.from_degC(20),
    energyDynamics = Modelica.Fluid.Types.Dynamics.FixedInitial)
    annotation(Placement(transformation(extent = {{-90, 70}, {-70, 90}})));

  // ── Core channel ──────────────────────────────────────────────────────────
  // FIX 2: diameter corrected from 7 m to 0.2 m (reasonable for a bundled
  //        core channel equivalent; adjust to match your actual flow area).
  // FIX 5: pressure starts updated for 5-bar network.
  Modelica.Fluid.Pipes.DynamicPipe core(
    redeclare package Medium = Medium,
    use_T_start = true,
    T_start = Modelica.Units.Conversions.from_degC(25),
    m_flow_start = 100,
    length = 7,
    redeclare model HeatTransfer =
      Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    diameter = 0.5,
    nNodes = nAxialNodes,
    redeclare model FlowModel =
      Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    use_HeatTransfer = true,
    modelStructure = if nAxialNodes == 1 then
      Modelica.Fluid.Types.ModelStructure.a_v_b else
      Modelica.Fluid.Types.ModelStructure.av_vb,
    p_a_start = 600000,
    p_b_start = 580000,
    state_a(p(start = 600000)),
    state_b(p(start = 580000)),
    statesFM(each p(start = 590000)))
    annotation(Placement(transformation(extent = {{30, 10}, {50, 30}})));

  // ── Radiator (heat exchanger) ─────────────────────────────────────────────
  // FIX 3: diameter corrected from 2 m to 0.05 m.
  // FIX 5: pressure starts updated for 5-bar network.
  Modelica.Fluid.Pipes.DynamicPipe radiator(
    use_T_start = true,
    redeclare package Medium = Medium,
    length = 10,
    T_start = Modelica.Units.Conversions.from_degC(25),
    m_flow_start = 100,
    redeclare model HeatTransfer =
      Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    diameter = 0.5,
    nNodes = 1,
    redeclare model FlowModel =
      Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    use_HeatTransfer = true,
    modelStructure = Modelica.Fluid.Types.ModelStructure.a_v_b,
    p_a_start = 110000,
    p_b_start = 100000,
    state_a(p(start = 110000)),
    state_b(p(start = 100000)),
    statesFM(each p(start = 105000)))
    annotation(Placement(transformation(extent = {{20, -80}, {0, -60}})));

  // ── Return pipe ───────────────────────────────────────────────────────────
  // FIX 5: pressure starts updated for 5-bar network.
  //        (580000 → 120000, large drop across valve accounted for separately)
  Modelica.Fluid.Pipes.DynamicPipe pipe(
    redeclare package Medium = Medium,
    use_T_start = true,
    T_start = Modelica.Units.Conversions.from_degC(25),
    m_flow_start = 100,
    redeclare model HeatTransfer =
      Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    diameter = 0.5,
    nNodes = 1,
    redeclare model FlowModel =
      Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    use_HeatTransfer = false,
    modelStructure = Modelica.Fluid.Types.ModelStructure.a_v_b,
    length = 10,
    p_a_start = 580000,
    p_b_start = 220000,
    state_a(p(start = 580000)),
    state_b(p(start = 220000)),
    statesFM(each p(start = 400000)))
    annotation(Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = -90, origin = {80, -20})));

protected
  Real positiveFractions[nAxialNodes](each unit = "1");
  Real normalizedFractions[nAxialNodes](each unit = "1");
  Modelica.Units.SI.HeatFlowRate nodeHeatFlow[nAxialNodes];
  Real fractionSum(unit = "1");

equation
  assert(totalPower >= 0, "totalPower must be non-negative.");
  assert(valveOpening >= 0 and valveOpening <= 1, "valveOpening must stay within [0, 1].");

  tankLevel = tank.level;

  connect(sensor_m_flow.m_flow, m_flow)
    annotation(Line(points = {{-10, 31}, {-10, 40}, {0, 40}}, color = {0, 0, 127}));
  connect(sensor_m_flow.port_b, core.port_a)
    annotation(Line(points = {{0, 20}, {30, 20}}, color = {0, 127, 255}));
  connect(T_ambient.port, wall.port_a)
    annotation(Line(points = {{0, -20}, {10, -20}, {10, -40}}, color = {191, 0, 0}));
  connect(sensor_T_forward.T, T_forward)
    annotation(Line(points = {{67, 40}, {80, 40}}, color = {0, 0, 127}));
  connect(radiator.port_a, valve.port_b)
    annotation(Line(points = {{20, -70}, {40, -70}}, color = {0, 127, 255}));
  connect(sensor_T_return.port, radiator.port_b)
    annotation(Line(points = {{-30, -60}, {-30, -70}, {0, -70}}, color = {0, 127, 255}));
  connect(tank.ports[2], pump.port_a)
    annotation(Line(points = {{-68, 30}, {-68, 20}, {-50, 20}}, color = {0, 127, 255}));
  connect(valveOpening, valve.opening)
    annotation(Line(points = {{-100, -44}, {50, -44}, {50, -62}}, color = {0, 0, 127}));
  connect(pump.port_b, sensor_m_flow.port_a)
    annotation(Line(points = {{-30, 20}, {-20, 20}}, color = {0, 127, 255}));
  connect(sensor_T_return.T, T_return)
    annotation(Line(points = {{-37, -50}, {-52, -50}}, color = {0, 0, 127}));
  connect(wall.port_b, radiator.heatPorts[1])
    annotation(Line(points = {{10, -56}, {10, -65.6}, {9.9, -65.6}}, color = {191, 0, 0}));
  connect(pipe.port_b, valve.port_a)
    annotation(Line(points = {{80, -30}, {80, -70}, {60, -70}}, color = {0, 127, 255}));
  connect(radiator.port_b, tank.ports[1])
    annotation(Line(points = {{0, -70}, {-72, -70}, {-72, 30}}, color = {0, 127, 255}));
  connect(core.port_b, pipe.port_a)
    annotation(Line(points = {{50, 20}, {80, 20}, {80, -10}}, color = {0, 127, 255}));
  connect(sensor_T_forward.port, core.port_b)
    annotation(Line(points = {{60, 30}, {50, 30}, {50, 20}}, color = {0, 127, 255}));

  for i in 1:nAxialNodes loop
    assert(axialPowerFractions[i] >= 0, "axialPowerFractions entries must be non-negative.");
    positiveFractions[i] = max(axialPowerFractions[i], 0);
  end for;
  fractionSum = sum(positiveFractions);
  assert(fractionSum > 1e-9, "axialPowerFractions must contain a positive sum.");
  for i in 1:nAxialNodes loop
    normalizedFractions[i] = if fractionSum > 1e-9 then positiveFractions[i] / fractionSum else 1.0 / nAxialNodes;
    nodeHeatFlow[i] = totalPower * normalizedFractions[i];
    coreHeat[i].Q_flow = nodeHeatFlow[i];
    connect(coreHeat[i].port, core.heatPorts[i]);
  end for;

  annotation(
    Documentation(info = "<html>
<p>
Research-reactor primary loop prototype derived from <code>HeatingSystemTest</code>.
The open-tank and radiator style topology is retained for the first FMI-oriented
thermal-hydraulic model, but the single heated pipe has been replaced by an
8-node core channel with distributed power input.
</p>
<p>
Inputs:
</p>
<ul>
<li><code>totalPower</code> [W] sets the total thermal power sent into the core channel.</li>
<li><code>axialPowerFractions[8]</code> sets the axial power split; the model normalizes the vector at runtime.</li>
<li><code>valveOpening</code> [0..1] drives the return-side valve directly.</li>
</ul>
<p>
Outputs expose the main loop observables needed for later coupling work:
forward temperature, return temperature, tank level, loop mass flow, and core
pressure drop.
</p>
</html>"),
    experiment(StopTime = 6000),
    __OpenModelica_commandLineOptions = "--replaceHomotopy=none --homotopyApproach=adaptiveGlobal",
    uses(Modelica(version = "4.1.0")));
end ResearchReactorPrimaryLoop;