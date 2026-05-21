model ResearchReactorThermalHydraulics
  "Reactor-scale D2O primary loop: 20 MW / 25 degC inlet / 45 degC outlet / 237 kg/s.
   Closed primary loop: inlet plenum -> core (8 nodes, top-to-bottom)
                        -> outlet plenum -> pump -> HX primary pipe -> inlet plenum.
  The reactor pool is represented as an open thermal reservoir coupled to the
  primary loop only through heat transfer; it is not hydraulically connected.
  Primary-loop pressure compliance is provided by a separate expansion tank.
   Secondary side: fixed-temperature heat sink at 20 degC, UA = 1.33 MW/K.
   Architecture reference: ReactorKineticsLab/theory/ThermalHydraulics.tex"

  parameter Integer nAxialNodes(min = 1) = 8
    "Axial nodes in the core channel";
  parameter Integer nHxNodes(min = 1) = 8
    "Axial nodes in the primary-side HX pipe";
  parameter Modelica.Units.SI.Power P_nominal = 20e6
    "Nominal thermal power [W]";
  parameter Modelica.Units.SI.MassFlowRate m_flow_nominal = 237
    "Nominal primary mass flow [kg/s]";
  parameter Modelica.Units.SI.Temperature T_in_nom = 298.15
    "Core inlet temperature at nominal [K]  (25 degC)";
  parameter Modelica.Units.SI.Temperature T_out_nom = 318.15
    "Core outlet temperature at nominal [K] (45 degC)";
  parameter Modelica.Units.SI.Temperature T_sec_fixed = 293.15
    "Secondary heat-sink temperature [K]    (20 degC)";
  parameter Modelica.Units.SI.ThermalConductance UA_hx = 1.333e6
    "Overall HX conductance [W/K]  ( = 20 MW / 15 K nominal primary-to-secondary delta-T )";
  parameter Modelica.Units.SI.Volume V_pool = 2535.0
    "Effective reactor-pool inventory represented as a thermal reservoir [m3]";
  parameter Modelica.Units.SI.Height poolHeight = 12.0
    "Geometric height of the open pool reservoir [m]";
  parameter Modelica.Units.SI.Height poolLevel_start = 10.0
    "Initial fluid level in the open pool reservoir [m]";
  parameter Modelica.Units.SI.Area poolCrossArea = V_pool / poolLevel_start
    "Cross-sectional area of the open pool reservoir [m2]";
  parameter Modelica.Units.SI.ThermalConductance G_pool_mix = 2.0e5
    "Lumped thermal coupling between inlet plenum and reactor pool [W/K]";
  parameter Modelica.Units.SI.ThermalConductance G_pool_ambient = 2.0e4
    "Lumped ambient heat-loss conductance from the open pool [W/K]";

  replaceable package Medium =
    Modelica.Media.CompressibleLiquids.LinearWater_pT_Ambient constrainedby
    Modelica.Media.Interfaces.PartialMedium
    "Primary coolant (water approximation for D2O; substitute once D2O package available)";

  // ── System ────────────────────────────────────────────────────────────────
  inner Modelica.Fluid.System system(
    p_start   = 4.0e5,
    T_start   = T_in_nom,
    m_flow_start = m_flow_nominal,
    m_flow_small = 1.0,
    energyDynamics = Modelica.Fluid.Types.Dynamics.FixedInitial,
    massDynamics = Modelica.Fluid.Types.Dynamics.DynamicFreeInitial)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-90, 70}, {-70, 90}})));

  // ── Inlet plenum (top of core; acts as closed-loop pressure reference) ────
  Modelica.Fluid.Vessels.ClosedVolume inletPlenum(
    redeclare package Medium = Medium,
    V           = 4.0,
    nPorts      = 3,
    use_portsData = false,
    use_HeatTransfer = true,
    p_start     = 4.0e5,
    T_start     = T_in_nom)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-40, 60}, {-20, 80}})));

  // ── Reactor pool (thermal reservoir only; no hydraulic coupling) ─────────
  Modelica.Fluid.Vessels.OpenTank poolInventory(
    redeclare package Medium = Medium,
    height      = poolHeight,
    crossArea   = poolCrossArea,
    level_start = poolLevel_start,
    nPorts      = 0,
    use_HeatTransfer = true,
    T_start     = T_in_nom,
    T_ambient   = system.T_ambient)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-98, 100}, {-74, 124}})));

  // ── Pressure reference for the primary loop ──────────────────────────────
  //   This keeps the cooling circuit independent of the pool while avoiding
  //   the unrealistically high absolute pressures produced by a fully sealed
  //   liquid-only loop model.
  Modelica.Fluid.Sources.Boundary_pT pressureBoundary(
    redeclare package Medium = Medium,
    nPorts      = 1,
    p           = system.p_start,
    T           = T_in_nom)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-72, 54}, {-52, 74}})));

  Modelica.Thermal.HeatTransfer.Components.ThermalConductor poolMixing(
    G = G_pool_mix)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-62, 88}, {-46, 104}})));

  Modelica.Thermal.HeatTransfer.Components.ThermalConductor poolAmbientLoss(
    G = G_pool_ambient)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-86, 72}, {-70, 88}})));

  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature ambientPoolBoundary(
    T = system.T_ambient)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-110, 70}, {-94, 86}})));

  // ── Effective core channel (top -> bottom, nAxialNodes cells) ─────────────
  //   A single axially-discretised pipe represents the entire active-core
  //   flow path.  Diameter and length are scaled so that the hydraulic
  //   residence time and heat capacity are representative of the annular
  //   fuel+inner-moderator region.  modelStructure = av_vb is required for
  //   nAxialNodes > 1 to avoid the regFun3 co-monotone interpolation failure.
  Modelica.Fluid.Pipes.DynamicPipe core(
    redeclare package Medium = Medium,
    use_T_start  = true,
    T_start      = T_in_nom,
    m_flow_start = m_flow_nominal,
    length       = 6.912,
    diameter     = 0.5,
    nNodes       = nAxialNodes,
    use_HeatTransfer = true,
    redeclare model HeatTransfer =
      Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    redeclare model FlowModel =
      Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    modelStructure = if nAxialNodes == 1 then
      Modelica.Fluid.Types.ModelStructure.a_v_b else
      Modelica.Fluid.Types.ModelStructure.av_vb,
    p_a_start  = 4.0e5,
    p_b_start  = 3.999e5,
    state_a(p(start = 4.0e5)),
    state_b(p(start = 3.999e5)),
    statesFM(each p(start = 3.9995e5)))
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-40, 20}, {-20, 40}})));

  // ── Outlet plenum (bottom of core) ────────────────────────────────────────
  Modelica.Fluid.Vessels.ClosedVolume outletPlenum(
    redeclare package Medium = Medium,
    V           = 4.0,
    nPorts      = 2,
    use_portsData = false,
    p_start     = 3.999e5,
    T_start     = T_out_nom)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-40, -20}, {-20, 0}})));

  Modelica.Fluid.Pipes.StaticPipe inletPipe(
    redeclare package Medium = Medium,
    length = 1.0,
    diameter = 0.5,
    p_a_start = 4.0e5,
    p_b_start = 4.0e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-40, 30}, {-20, 50}})));

  Modelica.Fluid.Pipes.StaticPipe outletPipe(
    redeclare package Medium = Medium,
    length = 1.0,
    diameter = 0.5,
    p_a_start = 3.999e5,
    p_b_start = 3.999e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-40, 0}, {-20, 20}})));

  Modelica.Fluid.Pipes.StaticPipe suctionPipe(
    redeclare package Medium = Medium,
    length = 1.0,
    diameter = 0.5,
    p_a_start = 3.999e5,
    p_b_start = 3.999e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-8, -20}, {10, 0}})));

  // ── Primary pump (flow-controlled: imposes m_flow = m_flow_nominal) ───────
  //   Use control_m_flow = true for first-pass thermal studies.
  //   Replace with head-curve pump for loss-of-flow / coastdown transients.
  Modelica.Fluid.Machines.ControlledPump pump(
    redeclare package Medium = Medium,
    N_nominal    = 1500,
    use_T_start  = true,
    T_start      = T_out_nom,
    m_flow_start = m_flow_nominal,
    m_flow_nominal = m_flow_nominal,
    control_m_flow = true,
    allowFlowReversal = false,
    p_a_start  = 3.999e5,
    p_b_start  = 4.005e5,
    p_a_nominal = 3.9e5,
    p_b_nominal = 4.1e5)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{10, -20}, {30, 0}})));

  // ── Mass-flow sensor (between pump outlet and HX inlet) ───────────────────
  Modelica.Fluid.Sensors.MassFlowRate sensor_m(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{40, -20}, {60, 0}})));

  Modelica.Fluid.Pipes.StaticPipe pumpToHxPipe(
    redeclare package Medium = Medium,
    length = 2.0,
    diameter = 0.4,
    p_a_start = 4.005e5,
    p_b_start = 4.005e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{60, 18}, {80, 38}})));

  // ── Primary-side HX pipe ──────────────────────────────────────────────────
  //   Single-node DynamicPipe with IdealFlowHeatTransfer.  Heat is removed
  //   through the single heatPorts[1] connection via hxWall -> T_sec.
  Modelica.Fluid.Pipes.DynamicPipe hxPipe(
    redeclare package Medium = Medium,
    use_T_start  = true,
    T_start      = 305.65,
    m_flow_start = m_flow_nominal,
    length       = 8.0,
    diameter     = 0.4,
    nNodes       = nHxNodes,
    use_HeatTransfer = true,
    redeclare model HeatTransfer =
      Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    redeclare model FlowModel =
      Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    modelStructure = if nHxNodes == 1 then
      Modelica.Fluid.Types.ModelStructure.a_v_b else
      Modelica.Fluid.Types.ModelStructure.av_vb,
    p_a_start  = 4.005e5,
    p_b_start  = 4.0e5,
    state_a(p(start = 4.005e5)),
    state_b(p(start = 4.0e5)),
    statesFM(each p(start = 4.0025e5)))
    annotation(Placement(transformation(origin = {0, -60}, extent = {{60, 54}, {80, 74}})));

  Modelica.Fluid.Pipes.StaticPipe returnPipe(
    redeclare package Medium = Medium,
    length = 2.0,
    diameter = 0.4,
    p_a_start = 4.0e5,
    p_b_start = 4.0e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{18, 60}, {38, 80}})));

  // ── HX thermal boundary (secondary side = fixed 20 degC heat sink) ────────
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor hxWall[nHxNodes](
    each G = UA_hx / nHxNodes)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{90, 60}, {110, 80}})));

  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature T_sec(
    T = T_sec_fixed)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{140, 60}, {120, 80}})));

  // ── Distributed core heat sources (driven by FMI input) ───────────────────
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow coreHeat[nAxialNodes]
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-90, 20}, {-70, 40}})));

  // ── Temperature sensors ───────────────────────────────────────────────────
  Modelica.Fluid.Sensors.Temperature sensor_T_in(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-80, 60}, {-60, 80}})));

  Modelica.Fluid.Sensors.Temperature sensor_T_out(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-80, -20}, {-60, 0}})));

  // ── Pressure sensors (for core pressure-drop output) ──────────────────────
  Modelica.Fluid.Sensors.Pressure sensor_p_in(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-10, 60}, {10, 80}})));

  Modelica.Fluid.Sensors.Pressure sensor_p_out(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-10, -20}, {10, 0}})));

  // ── FMI inputs ────────────────────────────────────────────────────────────
  Modelica.Blocks.Interfaces.RealInput totalPower(
    unit  = "W",
    min   = 0,
    start = P_nominal)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-130, 10}, {-110, 30}}), iconTransformation(extent = {{-130, 10}, {-110, 30}})));

  Modelica.Blocks.Interfaces.RealInput axialPowerFractions[nAxialNodes](
    each unit  = "1",
    each min   = 0,
    each start = 1.0 / nAxialNodes)
    annotation(Placement(transformation(origin = {0, -60}, extent = {{-130, -10}, {-110, 10}}), iconTransformation(extent = {{-130, -10}, {-110, 10}})));

  // ── FMI outputs ───────────────────────────────────────────────────────────
  Modelica.Blocks.Interfaces.RealOutput T_inlet(unit = "K")
    annotation(Placement(transformation(origin = {0, -60}, extent = {{110, 70}, {130, 90}}), iconTransformation(extent = {{110, 70}, {130, 90}})));

  Modelica.Blocks.Interfaces.RealOutput T_outlet(unit = "K")
    annotation(Placement(transformation(origin = {0, -60}, extent = {{110, 50}, {130, 70}}), iconTransformation(extent = {{110, 50}, {130, 70}})));

  Modelica.Blocks.Interfaces.RealOutput massFlow(unit = "kg/s")
    annotation(Placement(transformation(origin = {0, -60}, extent = {{110, 30}, {130, 50}}), iconTransformation(extent = {{110, 30}, {130, 50}})));

  Modelica.Blocks.Interfaces.RealOutput dp_core(unit = "Pa")
    annotation(Placement(transformation(origin = {0, -60}, extent = {{110, 10}, {130, 30}}), iconTransformation(extent = {{110, 10}, {130, 30}})));

protected
  Real positiveFractions[nAxialNodes](each unit = "1");
  Real normalizedFractions[nAxialNodes](each unit = "1");
  Modelica.Units.SI.HeatFlowRate nodeHeatFlow[nAxialNodes];
  Real fractionSum(unit = "1");

equation
  assert(totalPower >= 0, "totalPower must be non-negative");

  for i in 1:nAxialNodes loop
    assert(axialPowerFractions[i] >= 0,
      "axialPowerFractions entries must be non-negative");
    positiveFractions[i] = max(axialPowerFractions[i], 0);
  end for;
  fractionSum = sum(positiveFractions);
  assert(fractionSum > 1e-9,
    "axialPowerFractions must have at least one positive entry");

  for i in 1:nAxialNodes loop
    normalizedFractions[i] = if fractionSum > 1e-9 then
      positiveFractions[i] / fractionSum else 1.0 / nAxialNodes;
    nodeHeatFlow[i] = totalPower * normalizedFractions[i];
    coreHeat[i].Q_flow = nodeHeatFlow[i];
    connect(coreHeat[i].port, core.heatPorts[i]);
  end for;

  T_inlet  = sensor_T_in.T;
  T_outlet = sensor_T_out.T;
  massFlow = sensor_m.m_flow;
  dp_core  = sensor_p_in.p - sensor_p_out.p;

  // ── Primary loop fluid connections ────────────────────────────────────────
  connect(inletPlenum.ports[1], inletPipe.port_a)
    annotation(Line(points = {{-30, 0}, {-30, -10}}, color = {0, 127, 255}));
  connect(inletPipe.port_b, core.port_a)
    annotation(Line(points = {{-20, -20}, {-20, -20}, {-20, -30}}, color = {0, 127, 255}));
  connect(core.port_b, outletPipe.port_a)
    annotation(Line(points = {{-20, -30}, {-20, -40}}, color = {0, 127, 255}));
  connect(outletPipe.port_b, outletPlenum.ports[1])
    annotation(Line(points = {{-20, -50}, {-20, -60}}, color = {0, 127, 255}));
  connect(outletPlenum.ports[2], suctionPipe.port_a)
    annotation(Line(points = {{-30, -80}, {-30, -70}, {-8, -70}}, color = {0, 127, 255}));
  connect(suctionPipe.port_b, pump.port_a)
    annotation(Line(points = {{10, -70}, {10, -70}}, color = {0, 127, 255}));
  connect(pump.port_b, sensor_m.port_a)
    annotation(Line(points = {{30, -70}, {40, -70}}, color = {0, 127, 255}));
  connect(sensor_m.port_b, pumpToHxPipe.port_a)
    annotation(Line(points = {{60, -70}, {70, -70}, {70, -42}}, color = {0, 127, 255}));
  connect(pumpToHxPipe.port_b, hxPipe.port_a)
    annotation(Line(points = {{80, -32}, {90, -32}, {90, 4}, {80, 4}}, color = {0, 127, 255}));
  connect(hxPipe.port_b, returnPipe.port_a)
    annotation(Line(points = {{80, 4}, {70, 4}, {70, 10}, {58, 10}}, color = {0, 127, 255}));
  connect(returnPipe.port_b, inletPlenum.ports[2])
    annotation(Line(points = {{38, 10}, {10, 10}, {10, 20}, {-10, 20}}, color = {0, 127, 255}));
  connect(inletPlenum.ports[3], pressureBoundary.ports[1])
    annotation(Line(points = {{-30, 0}, {-30, 12}, {-62, 12}, {-62, -26}}, color = {0, 127, 255}));

  // ── Sensor port taps (zero-flow connections to existing fluid nodes) ───────
  connect(sensor_T_in.port, core.port_a)
    annotation(Line(points = {{-70, 0}, {-60, 0}, {-60, -40}, {-30, -40}}, color = {0, 127, 255}));
  connect(sensor_T_out.port, core.port_b)
    annotation(Line(points = {{-70, -80}, {-60, -80}, {-60, -40}, {-30, -40}}, color = {0, 127, 255}));
  connect(sensor_p_in.port, core.port_a)
    annotation(Line(points = {{0, 0}, {20, 0}, {20, -40}, {-20, -40}}, color = {0, 127, 255}));
  connect(sensor_p_out.port, core.port_b)
    annotation(Line(points = {{0, -80}, {20, -80}, {20, -40}, {-20, -40}}, color = {0, 127, 255}));

  connect(inletPlenum.heatPort, poolMixing.port_a)
    annotation(Line(points = {{-40, 10}, {-40, 36}, {-72, 36}}, color = {191, 0, 0}));
  connect(poolMixing.port_b, poolInventory.heatPort)
    annotation(Line(points = {{-46, 36}, {-46, 52}, {-74, 52}}, color = {191, 0, 0}));
  connect(poolInventory.heatPort, poolAmbientLoss.port_a)
    annotation(Line(points = {{-98, 52}, {-110, 52}, {-110, 20}}, color = {191, 0, 0}));
  connect(poolAmbientLoss.port_b, ambientPoolBoundary.port)
    annotation(Line(points = {{-70, 20}, {-94, 20}}, color = {191, 0, 0}));

  // ── HX thermal path ───────────────────────────────────────────────────────
  for i in 1:nHxNodes loop
    connect(hxPipe.heatPorts[i], hxWall[i].port_a);
    connect(hxWall[i].port_b, T_sec.port);
  end for;

  annotation(
    Documentation(info = "<html>
<p>
Reactor-scale thermal-hydraulic model for the Reactor Kinetics Lab (20&nbsp;MW,
D<sub>2</sub>O primary loop).  Architecture specified in
<code>ReactorKineticsLab/theory/ThermalHydraulics.tex</code>.
</p>
<ul>
<li>Closed D<sub>2</sub>O primary loop: inlet plenum &rarr; inlet pipe &rarr;
  core (top-to-bottom, 8 axial nodes) &rarr; outlet pipe &rarr; outlet plenum
  &rarr; suction pipe &rarr; pump &rarr; mass-flow sensor &rarr; HX inlet pipe
  &rarr; distributed HX primary pipe &rarr; return pipe &rarr; inlet plenum.</li>
<li>The reactor pool is modeled as an open thermal reservoir coupled to the
  inlet plenum only through heat transfer. It is not hydraulically connected
  to the primary loop.</li>
<li>A separate fixed-pressure boundary provides the primary-loop pressure
  reference without conflating the pool with the sealed cooling circuit.</li>
<li>Design operating point: 25&nbsp;&deg;C inlet, 45&nbsp;&deg;C outlet,
  237&nbsp;kg/s, ~4&nbsp;bar absolute system pressure.</li>
<li>Flow-controlled pump (constant 237&nbsp;kg/s).  Replace
    <code>control_m_flow = true</code> with a head-curve pump for
    loss-of-flow transient studies.</li>
<li>Secondary side: fixed 20&nbsp;&deg;C thermal boundary with
  UA = 1.333&nbsp;MW/K distributed across <code>nHxNodes</code> cells.</li>
</ul>
<p>
<b>FMI inputs:</b> <code>totalPower</code>&nbsp;[W],
<code>axialPowerFractions[n]</code><br/>
<b>FMI outputs:</b> <code>T_inlet</code>&nbsp;[K],
<code>T_outlet</code>&nbsp;[K], <code>massFlow</code>&nbsp;[kg/s],
<code>dp_core</code>&nbsp;[Pa]
</p>
</html>"),
    experiment(StopTime = 6000, Interval = 5.0, Tolerance = 1e-6),
    uses(Modelica(version = "4.1.0")));

end ResearchReactorThermalHydraulics;