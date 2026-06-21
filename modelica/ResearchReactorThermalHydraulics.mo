model ResearchReactorThermalHydraulics
  "Reactor-scale D2O primary loop: 20 MW / 25 degC inlet / 45 degC outlet / 237 kg/s.
   Closed primary loop: inlet plenum -> core (8 nodes, top-to-bottom)
                        -> outlet plenum -> pump -> HX primary pipe -> inlet plenum.
  The reactor pool is represented as an open thermal reservoir coupled to the
  primary loop only through heat transfer; it is not hydraulically connected.
  The inlet plenum is modeled as a closed compliance volume with a fixed loop pressure reference.
   Secondary side: fixed-temperature heat sink at 20 degC, UA = 1.33 MW/K.
   Architecture reference: ReactorKineticsLab/theory/ThermalHydraulics.tex"

  function rhoD2O_pT
    input Modelica.Units.SI.AbsolutePressure p;
    input Modelica.Units.SI.Temperature T;
    output Modelica.Units.SI.Density d;
  protected
    parameter Modelica.Units.SI.Density d_ref = 1105.0;
    parameter Modelica.Units.SI.Temperature T_ref = 300.0;
    parameter Modelica.Units.SI.AbsolutePressure p_ref = 4.0e5;
    parameter Real beta(unit = "1/K") = 3.0e-4;
    parameter Real kappa(unit = "1/Pa") = 4.5e-10;
  algorithm
    d := d_ref * (1 - beta * (T - T_ref) + kappa * (p - p_ref));
  end rhoD2O_pT;

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
  parameter Modelica.Units.SI.Temperature T_core_in_start = 300.675
    "Nominal steady-state core inlet temperature used for initialization [K]";
  parameter Modelica.Units.SI.Temperature T_core_out_start = 320.854
    "Nominal steady-state core outlet temperature used for initialization [K]";
  final parameter Modelica.Units.SI.Temperature T_core_avg_start =
    0.5 * (T_core_in_start + T_core_out_start)
    "Nominal steady-state core-average coolant temperature used for initialization [K]";
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
  parameter Modelica.Units.SI.AbsolutePressure p_loop_ref = 4.0e5
    "Primary-loop absolute pressure reference used to anchor the FMU pressure level [Pa]";
  parameter Integer nFuelRadialNodes(min = 2) = 5
    "Radial nodes in the fuel-meat half-slab heat structure";
  parameter Integer fuelRingCount(min = 1) = 6
    "OpenMC concentric fuel-ring count used for aggregated slab volume";
  parameter Modelica.Units.SI.Length fuelRingThickness = 0.005
    "OpenMC concentric fuel-ring thickness [m]";
  parameter Modelica.Units.SI.Length fuelCoolantGap = 0.07
    "OpenMC concentric coolant gap between fuel rings [m]";
  parameter Modelica.Units.SI.Length fuelInnerRadius = 0.045
    "OpenMC concentric element inner radius [m]";
  parameter Modelica.Units.SI.Length fuelOuterRadius = 0.50
    "OpenMC concentric element outer radius [m]";
  parameter Modelica.Units.SI.Length fuelActiveHeight = 1.5
    "OpenMC concentric active fuel height [m]";
  parameter Modelica.Units.SI.Density fuelDensity = 12200
    "Fuel meat density used for heat-structure capacity [kg/m3]";
  parameter Modelica.Units.SI.SpecificHeatCapacity fuelSpecificHeatCapacity = 116
    "Fuel meat specific heat capacity [J/(kg.K)]";
  parameter Modelica.Units.SI.ThermalConductivity fuelThermalConductivity = 27
    "Fuel meat thermal conductivity [W/(m.K)]";
  parameter Modelica.Units.SI.CoefficientOfHeatTransfer h_core = 2.0e4
    "Fixed fuel-wall to coolant heat-transfer coefficient [W/(m2.K)]";
  parameter Modelica.Units.SI.Length coreHydraulicDiameter = 0.5
    "Effective single-channel hydraulic diameter retained for loop pressure-drop calibration [m]";
  parameter Modelica.Units.SI.Temperature T_fuel_start =
    T_core_avg_start +
    (P_nominal / nAxialNodes / fuelAxialNodeArea) *
    (1 / h_core + fuelHalfThickness / (3 * fuelThermalConductivity))
    "Nominal steady-state fuel-average temperature used for initialization [K]";

  final parameter Modelica.Units.SI.Length fuelRadialSpan =
    fuelRingCount * fuelRingThickness + (fuelRingCount - 1) * fuelCoolantGap
    "Total radial span occupied by concentric fuel rings and coolant gaps";
  final parameter Modelica.Units.SI.Length fuelEdgeGap =
    0.5 * ((fuelOuterRadius - fuelInnerRadius) - fuelRadialSpan)
    "Unused radial edge gap in the OpenMC concentric element";
  final parameter Modelica.Units.SI.Radius fuelRingInnerRadii[fuelRingCount] =
    {fuelInnerRadius + fuelEdgeGap + (i - 1) * (fuelRingThickness + fuelCoolantGap)
      for i in 1:fuelRingCount}
    "Inner radii of the concentric fuel rings";
  final parameter Modelica.Units.SI.Area fuelCrossSectionArea =
    Modelica.Constants.pi *
    sum((fuelRingInnerRadii[i] + fuelRingThickness)^2 - fuelRingInnerRadii[i]^2
      for i in 1:fuelRingCount)
    "Total concentric fuel-meat cross-section area";
  final parameter Modelica.Units.SI.Volume fuelVolume =
    fuelCrossSectionArea * fuelActiveHeight
    "Aggregated fuel-meat volume represented by the slab";
  final parameter Modelica.Units.SI.Mass fuelMass =
    fuelDensity * fuelVolume
    "Aggregated fuel-meat mass represented by the slab";
  final parameter Modelica.Units.SI.Length fuelHalfThickness =
    0.5 * fuelRingThickness
    "Fuel-meat half thickness from symmetry plane to coolant interface";
  final parameter Modelica.Units.SI.Length fuelRadialCellWidth =
    fuelHalfThickness / nFuelRadialNodes
    "Radial width of each solid finite-volume node";
  final parameter Modelica.Units.SI.Area fuelHeatTransferArea =
    fuelVolume / fuelHalfThickness
    "Two-sided slab-equivalent fuel heat-transfer area";
  final parameter Modelica.Units.SI.Area fuelAxialNodeArea =
    fuelHeatTransferArea / nAxialNodes
    "Fuel heat-transfer area assigned to each axial coolant node";
  final parameter Modelica.Units.SI.Volume fuelNodeVolume =
    fuelVolume / (nAxialNodes * nFuelRadialNodes)
    "Fuel volume assigned to each axial/radial solid node";
  final parameter Modelica.Units.SI.HeatCapacity fuelNodeHeatCapacity =
    fuelDensity * fuelSpecificHeatCapacity * fuelNodeVolume
    "Heat capacity assigned to each solid node";
  final parameter Modelica.Units.SI.ThermalConductance fuelInternalConductance =
    fuelThermalConductivity * fuelAxialNodeArea / fuelRadialCellWidth
    "Conductance between adjacent radial fuel-node centers";
  final parameter Modelica.Units.SI.ThermalConductance fuelWallConductance =
    2 * fuelThermalConductivity * fuelAxialNodeArea / fuelRadialCellWidth
    "Half-cell conductance from the outer fuel-node center to the wall";
  final parameter Modelica.Units.SI.ThermalConductance fuelConvectiveConductance =
    h_core * fuelAxialNodeArea
    "Fuel wall-to-coolant convective conductance per axial node";

  replaceable package Medium =
    Modelica.Media.CompressibleLiquids.LinearWater_pT_Ambient constrainedby
    Modelica.Media.Interfaces.PartialMedium
    "Primary coolant (water approximation for D2O; substitute once D2O package available)";

// ── System ────────────────────────────────────────────────────────────────
  inner Modelica.Fluid.System system(
    p_start   = 4.0e5,
    T_start   = T_core_in_start,
    m_flow_start = m_flow_nominal,
    m_flow_small = 1.0,
    energyDynamics = Modelica.Fluid.Types.Dynamics.FixedInitial,
    massDynamics = Modelica.Fluid.Types.Dynamics.DynamicFreeInitial)
    annotation(Placement(transformation(origin = {150, 4}, extent = {{-90, 70}, {-70, 90}})));

// ── Inlet plenum (top of core pressure state) ────────────────────────────
  Modelica.Fluid.Vessels.ClosedVolume inletPlenum(
    redeclare package Medium = Medium,
    V           = 4.0,
    nPorts      = 3,
    use_portsData = false,
    use_HeatTransfer = true,
    p_start     = p_loop_ref,
    T_start     = T_core_in_start,
    ports_H_flow(each min = -2.0e8, each max = 2.0e8))
    annotation(Placement(transformation(origin = {8, 26}, extent = {{-24, 36}, {-12, 48}})));

  Modelica.Fluid.Sources.Boundary_pT pressureReference(
    redeclare package Medium = Medium,
    nPorts = 1,
    p = p_loop_ref,
    T = T_core_in_start)
    annotation(Placement(transformation(origin = {-34, 80}, extent = {{-12, -10}, {8, 10}})));

// ── Reactor pool (thermal reservoir only; no hydraulic coupling) ─────────
  Modelica.Fluid.Vessels.OpenTank poolInventory(
    redeclare package Medium = Medium,
    height      = poolHeight,
    crossArea   = poolCrossArea,
    level_start = poolLevel_start,
    nPorts      = 0,
    use_HeatTransfer = true,
    T_start     = T_core_in_start,
    T_ambient   = system.T_ambient)
    annotation(Placement(transformation(origin = {64.8333, -101.667}, extent = {{-138.833, 141.667}, {-104.833, 175.667}})));

  Modelica.Thermal.HeatTransfer.Components.ThermalConductor poolMixing(
    G = G_pool_mix)
    annotation(Placement(transformation(origin = {4.75, 29}, extent = {{-38.75, 55}, {-28.75, 65}})));

  Modelica.Thermal.HeatTransfer.Components.ThermalConductor poolAmbientLoss(
    G = G_pool_ambient)
    annotation(Placement(transformation(origin = {-16.75, -35}, extent = {{-75.25, 63}, {-61.25, 77}})));

  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature ambientPoolBoundary(
    T = system.T_ambient)
    annotation(Placement(transformation(origin = {14, -70}, extent = {{-110, 70}, {-94, 86}})));

// ── Effective core channel (top -> bottom, nAxialNodes cells) ─────────────
  //   A single axially-discretised pipe represents the entire active-core
  //   flow path.  The active length follows the OpenMC concentric fuel element;
  //   the diameter remains an effective hydraulic calibration parameter rather
  //   than the 1.2 m fuel-envelope outer diameter.
  //   modelStructure = av_vb is required for
  //   nAxialNodes > 1 to avoid the regFun3 co-monotone interpolation failure.
  Modelica.Fluid.Pipes.DynamicPipe core(
    redeclare package Medium = Medium,
    use_T_start  = true,
    T_start      = T_core_avg_start,
    m_flow_start = m_flow_nominal,
    length       = fuelActiveHeight,
    diameter     = coreHydraulicDiameter,
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
    statesFM(each p(start = 3.9995e5)),
    H_flows(each min = -2.0e8, each max = 2.0e8))
    annotation(Placement(transformation(origin = {-66, -60}, extent = {{-76, 38}, {-38, 76}}, rotation = -90)));

// ── Outlet plenum (bottom of core) ────────────────────────────────────────
  Modelica.Fluid.Vessels.ClosedVolume outletPlenum(
    redeclare package Medium = Medium,
    V           = 4.0,
    nPorts      = 2,
    use_portsData = false,
    p_start     = 3.999e5,
    T_start     = T_core_out_start)
    annotation(Placement(transformation(origin = {12, -84}, extent = {{-28, 14}, {-14, -0}}, rotation = -0)));

  Modelica.Fluid.Pipes.StaticPipe inletPipe(
    redeclare package Medium = Medium,
    length = 1.0,
    diameter = 0.5,
    p_a_start = 4.0e5,
    p_b_start = 4.0e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {-52.6433, 16.5017}, extent = {{-38.1207, 32.304}, {-19.0604, 53.84}}, rotation = -90)));

  Modelica.Fluid.Pipes.StaticPipe outletPipe(
    redeclare package Medium = Medium,
    length = 1.0,
    diameter = 0.5,
    p_a_start = 3.999e5,
    p_b_start = 3.999e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {-20, -78}, extent = {{-44, 0}, {-22, 22}}, rotation = -90)));

  Modelica.Fluid.Pipes.StaticPipe suctionPipe(
    redeclare package Medium = Medium,
    length = 1.0,
    diameter = 0.5,
    p_a_start = 3.999e5,
    p_b_start = 3.999e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {14, -60}, extent = {{-8, -20}, {10, 0}})));

// ── Primary pump (flow-controlled: imposes m_flow = m_flow_nominal) ───────
  //   Use control_m_flow = true for first-pass thermal studies.
  //   Replace with head-curve pump for loss-of-flow / coastdown transients.
  Modelica.Fluid.Machines.ControlledPump pump(
    redeclare package Medium = Medium,
    N_nominal    = 1500,
    use_T_start  = true,
    T_start      = T_core_out_start,
    m_flow_start = m_flow_nominal,
    m_flow_nominal = m_flow_nominal,
    control_m_flow = true,
    allowFlowReversal = false,
    p_a_start  = 3.999e5,
    p_b_start  = 4.005e5,
    p_a_nominal = 3.9e5,
    p_b_nominal = 4.1e5)
    annotation(Placement(transformation(origin = {18, -60}, extent = {{10, -20}, {30, 0}})));

// ── Mass-flow sensor (between pump outlet and HX inlet) ───────────────────
  Modelica.Fluid.Sensors.MassFlowRate sensor_m(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {46, -83.1921}, extent = {{24.284, -11.8911}, {36.4261, 0}}, rotation = 90)));

  Modelica.Fluid.Pipes.StaticPipe pumpToHxPipe(
    redeclare package Medium = Medium,
    length = 2.0,
    diameter = 0.4,
    p_a_start = 4.005e5,
    p_b_start = 4.005e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {80, -96}, extent = {{60, 18}, {80, 38}}, rotation = 90)));

// ── Primary-side HX pipe ──────────────────────────────────────────────────
  //   Single-node DynamicPipe with IdealFlowHeatTransfer.  Heat is removed
  //   through the single heatPorts[1] connection via hxWall -> T_sec.
  Modelica.Fluid.Pipes.DynamicPipe hxPipe(
    redeclare package Medium = Medium,
    use_T_start  = true,
    T_start      = T_core_avg_start,
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
    annotation(Placement(transformation(origin = {-37.6, -76}, extent = {{-84, 75.6}, {-112, 103.6}}, rotation = -90)));

  Modelica.Fluid.Pipes.StaticPipe returnPipe(
    redeclare package Medium = Medium,
    length = 2.0,
    diameter = 0.4,
    p_a_start = 4.0e5,
    p_b_start = 4.0e5,
    m_flow_start = m_flow_nominal)
    annotation(Placement(transformation(origin = {52, 132}, extent = {{18, 60}, {38, 80}}, rotation = 180)));

// ── HX thermal boundary (secondary side = fixed 20 degC heat sink) ────────
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor hxWall[nHxNodes](
    each G = UA_hx / nHxNodes)
    annotation(Placement(transformation(origin = {-26, -48}, extent = {{90, 60}, {110, 80}})));

  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature T_sec(
    T = T_sec_fixed)
    annotation(Placement(transformation(origin = {190, 112}, extent = {{126, 54}, {108, 72}}, rotation = 180)));

// ── RELAP-style fuel heat structure (half-slab -> coolant bulk) ───────────
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor fuelNode[nAxialNodes, nFuelRadialNodes](
    each C = fuelNodeHeatCapacity,
    each T(start = T_fuel_start));

  Modelica.Thermal.HeatTransfer.Components.ThermalConductor fuelRadialConduction[
    nAxialNodes, nFuelRadialNodes - 1](
    each G = fuelInternalConductance);

  Modelica.Thermal.HeatTransfer.Components.ThermalConductor fuelWallConduction[nAxialNodes](
    each G = fuelWallConductance);

  Modelica.Thermal.HeatTransfer.Components.Convection coreConvection[nAxialNodes];

  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow fuelHeat[
    nAxialNodes, nFuelRadialNodes];

// ── Temperature sensors ───────────────────────────────────────────────────
  Modelica.Fluid.Sensors.Temperature sensor_T_in(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {8, -8}, extent = {{-48, 30}, {-36, 40}})));

  Modelica.Fluid.Sensors.Temperature sensor_T_out(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {8, -20}, extent = {{-48, -10}, {-36, 0}})));

// ── Pressure sensors (for core pressure-drop output) ──────────────────────
  Modelica.Fluid.Sensors.Pressure sensor_p_in(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {18, -14}, extent = {{-6, 36}, {6, 48}})));

  Modelica.Fluid.Sensors.Pressure sensor_p_out(
    redeclare package Medium = Medium)
    annotation(Placement(transformation(origin = {18, -20}, extent = {{-6, -12}, {6, 0}})));

// ── FMI inputs ────────────────────────────────────────────────────────────
  Modelica.Blocks.Interfaces.RealInput totalPower(
    unit  = "W",
    min   = 0,
    start = P_nominal)
    annotation(Placement(transformation(origin = {4, -12}, extent = {{-130, 10}, {-110, 30}}), iconTransformation(extent = {{-130, 10}, {-110, 30}})));

  Modelica.Blocks.Interfaces.RealInput axialPowerFractions[nAxialNodes](
    each unit  = "1",
    each min   = 0,
    each start = 1.0 / nAxialNodes)
    annotation(Placement(transformation(origin = {4, -16}, extent = {{-130, -10}, {-110, 10}}), iconTransformation(extent = {{-130, -10}, {-110, 10}})));

// ── FMI outputs ───────────────────────────────────────────────────────────
  Modelica.Blocks.Interfaces.RealOutput T_inlet(unit = "K")
    annotation(Placement(transformation(origin = {-8, 8}, extent = {{110, 70}, {130, 90}}), iconTransformation(extent = {{110, 70}, {130, 90}})));

  Modelica.Blocks.Interfaces.RealOutput T_outlet(unit = "K")
    annotation(Placement(transformation(origin = {-8, 8}, extent = {{110, 50}, {130, 70}}), iconTransformation(extent = {{110, 50}, {130, 70}})));

  Modelica.Blocks.Interfaces.RealOutput massFlow(unit = "kg/s")
    annotation(Placement(transformation(origin = {-8, 8}, extent = {{110, 30}, {130, 50}}), iconTransformation(extent = {{110, 30}, {130, 50}})));

  Modelica.Blocks.Interfaces.RealOutput dp_core(unit = "Pa")
    annotation(Placement(transformation(origin = {-8, 6}, extent = {{110, 10}, {130, 30}}), iconTransformation(extent = {{110, 10}, {130, 30}})));

  Modelica.Blocks.Interfaces.RealOutput T_fuelCenterlineMax(unit = "K")
    annotation(Placement(transformation(origin = {-8, 6}, extent = {{110, -10}, {130, 10}}), iconTransformation(extent = {{110, -10}, {130, 10}})));

  Modelica.Blocks.Interfaces.RealOutput T_fuelWallMax(unit = "K")
    annotation(Placement(transformation(origin = {-8, 6}, extent = {{110, -30}, {130, -10}}), iconTransformation(extent = {{110, -30}, {130, -10}})));

  Modelica.Blocks.Interfaces.RealOutput T_fuelEff(unit = "K")
    annotation(Placement(transformation(origin = {-8, 6}, extent = {{110, -50}, {130, -30}}), iconTransformation(extent = {{110, -50}, {130, -30}})));

  Modelica.Blocks.Interfaces.RealOutput T_moderatorEff(unit = "K")
    annotation(Placement(transformation(origin = {-8, 6}, extent = {{110, -70}, {130, -50}}), iconTransformation(extent = {{110, -70}, {130, -50}})));

  Modelica.Blocks.Interfaces.RealOutput rho_m_eff_SI(unit = "kg/m3")
    annotation(Placement(transformation(origin = {-8, 6}, extent = {{110, -90}, {130, -70}}), iconTransformation(extent = {{110, -90}, {130, -70}})));

  Modelica.Blocks.Interfaces.RealOutput rho_m_eff(unit = "g/cm3")
    annotation(Placement(transformation(origin = {-8, 6}, extent = {{110, -110}, {130, -90}}), iconTransformation(extent = {{110, -110}, {130, -90}})));


protected
  Real positiveFractions[nAxialNodes](each unit = "1");
  Real normalizedFractions[nAxialNodes](each unit = "1");
  Modelica.Units.SI.HeatFlowRate nodeHeatFlow[nAxialNodes];
  Modelica.Units.SI.AbsolutePressure p_coreNode[nAxialNodes];
  Modelica.Units.SI.Temperature T_coreNode[nAxialNodes];
  Real fractionSum(unit = "1");

equation
  assert(totalPower >= 0, "totalPower must be non-negative");
  assert(fuelOuterRadius > fuelInnerRadius,
    "fuelOuterRadius must be larger than fuelInnerRadius");
  assert(fuelEdgeGap >= 0,
    "OpenMC concentric fuel rings do not fit between fuelInnerRadius and fuelOuterRadius");

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

    for r in 1:nFuelRadialNodes loop
      fuelHeat[i, r].Q_flow = nodeHeatFlow[i] / nFuelRadialNodes;
      connect(fuelHeat[i, r].port, fuelNode[i, r].port);
    end for;
    // No inward connection on radial node 1: zero heat flux at the symmetry plane.
    for r in 1:(nFuelRadialNodes - 1) loop
      connect(fuelNode[i, r].port, fuelRadialConduction[i, r].port_a);
      connect(fuelRadialConduction[i, r].port_b, fuelNode[i, r + 1].port);
    end for;

    connect(fuelNode[i, nFuelRadialNodes].port, fuelWallConduction[i].port_a);
    connect(fuelWallConduction[i].port_b, coreConvection[i].solid);
    coreConvection[i].Gc = fuelConvectiveConductance;
    connect(coreConvection[i].fluid, core.heatPorts[i]);
  end for;

  T_inlet  = sensor_T_in.T;
  T_outlet = sensor_T_out.T;
  massFlow = sensor_m.m_flow;
  dp_core  = sensor_p_in.p - sensor_p_out.p;
  T_fuelCenterlineMax = max({fuelNode[i, 1].T for i in 1:nAxialNodes});
  T_fuelWallMax = max({fuelWallConduction[i].port_b.T for i in 1:nAxialNodes});
  T_fuelEff = sum({normalizedFractions[i] *
    sum({fuelNode[i, r].T for r in 1:nFuelRadialNodes}) / nFuelRadialNodes
    for i in 1:nAxialNodes});
  for i in 1:nAxialNodes loop
    p_coreNode[i] = sensor_p_in.p -
      ((i - 0.5) / nAxialNodes) * (sensor_p_in.p - sensor_p_out.p);
    T_coreNode[i] = core.heatPorts[i].T;
  end for;
  T_moderatorEff = sum({normalizedFractions[i] * T_coreNode[i]
    for i in 1:nAxialNodes});
  rho_m_eff_SI = sum({normalizedFractions[i] * rhoD2O_pT(p_coreNode[i], T_coreNode[i])
    for i in 1:nAxialNodes});
  rho_m_eff = rho_m_eff_SI / 1000.0;
// ── Primary loop fluid connections ────────────────────────────────────────
  connect(core.port_b, outletPipe.port_a) annotation(
    Line(points = {{-9, -22}, {-9, -34}}, color = {0, 127, 255}));
  connect(outletPipe.port_b, outletPlenum.ports[1])
    annotation(Line(points = {{-9, -56}, {-9, -70}}, color = {0, 127, 255}));
  connect(outletPlenum.ports[2], suctionPipe.port_a)
    annotation(Line(points = {{-9, -70}, {6, -70}}, color = {0, 127, 255}));
  connect(suctionPipe.port_b, pump.port_a)
    annotation(Line(points = {{24, -70}, {28, -70}}, color = {0, 127, 255}));
  connect(pumpToHxPipe.port_b, hxPipe.port_a)
    annotation(Line(points = {{52, -16}, {51.625, -16}, {51.625, 8}, {52, 8}}, color = {0, 127, 255}));
  connect(hxPipe.port_b, returnPipe.port_a)
    annotation(Line(points = {{52, 36}, {52, 62}, {34, 62}}, color = {0, 127, 255}));
  connect(returnPipe.port_b, inletPlenum.ports[1]) annotation(
    Line(points = {{14, 62}, {-10, 62}}, color = {0, 127, 255}));
  connect(pressureReference.ports[1], inletPlenum.ports[3]) annotation(
    Line(points = {{-26, 80}, {-10, 80}, {-10, 62}}, color = {0, 127, 255}));
// ── Sensor port taps (zero-flow connections to existing fluid nodes) ───────
  connect(poolInventory.heatPort, poolAmbientLoss.port_a) annotation(
    Line(points = {{-74, 57}, {-96, 57}, {-96, 35}, {-92, 35}}, color = {191, 0, 0}));
  connect(poolAmbientLoss.port_b, ambientPoolBoundary.port)
    annotation(Line(points = {{-78, 35}, {-78, 35.5}, {-70, 35.5}, {-70, 7.375}, {-80, 7.375}, {-80, 8}}, color = {191, 0, 0}));
// ── HX thermal path ───────────────────────────────────────────────────────
  for i in 1:nHxNodes loop
    connect(hxPipe.heatPorts[i], hxWall[i].port_a);
    connect(hxWall[i].port_b, T_sec.port);
  end for;
  connect(poolMixing.port_a, poolInventory.heatPort) annotation(
    Line(points = {{-34, 89}, {-74, 89}, {-74, 57}}, color = {191, 0, 0}));
  connect(inletPipe.port_a, inletPlenum.ports[2]) annotation(
    Line(points = {{-10, 55}, {-10, 62}}, color = {0, 127, 255}));
  connect(inletPlenum.heatPort, poolMixing.port_b) annotation(
    Line(points = {{-16, 68}, {-20, 68}, {-20, 89}, {-24, 89}}, color = {191, 0, 0}));
  connect(sensor_T_out.port, core.port_b) annotation(
    Line(points = {{-34, -30}, {-34, -32}, {-8, -32}, {-8, -22}}, color = {0, 127, 255}));
  connect(sensor_p_out.port, core.port_b) annotation(
    Line(points = {{18, -32}, {-8, -32}, {-8, -22}}, color = {0, 127, 255}));
  connect(inletPipe.port_b, core.port_a) annotation(
    Line(points = {{-10, 36}, {-10, 15.5}, {-8, 15.5}, {-8, 16}}, color = {0, 127, 255}));
  connect(sensor_T_in.port, core.port_a) annotation(
    Line(points = {{-34, 22}, {-34, 18}, {-8, 18}, {-8, 16}}, color = {0, 127, 255}));
  connect(sensor_p_in.port, core.port_a) annotation(
    Line(points = {{18, 22}, {18, 18}, {-8, 18}, {-8, 16}}, color = {0, 127, 255}));
  connect(pump.port_b, sensor_m.port_a) annotation(
    Line(points = {{48, -70}, {52, -70}, {52, -59}}, color = {0, 127, 255}));
  connect(sensor_m.port_b, pumpToHxPipe.port_a) annotation(
    Line(points = {{52, -46}, {52, -36}}, color = {0, 127, 255}));
  connect(hxWall.port_a, hxPipe.heatPorts) annotation(
    Line(points = {{64, 22}, {58, 22}}, color = {191, 0, 0}, thickness = 0.5));
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
<li>The inlet plenum is a closed compliance volume with a fixed 4&nbsp;bar
  pressure-reference boundary. This anchors the FMU pressure level without
  hydraulically connecting the pool.</li>
<li>Design operating point: 25&nbsp;&deg;C inlet, 45&nbsp;&deg;C outlet,
  237&nbsp;kg/s, ~4&nbsp;bar absolute system pressure.</li>
<li>Flow-controlled pump (constant 237&nbsp;kg/s).  Replace
    <code>control_m_flow = true</code> with a head-curve pump for
    loss-of-flow transient studies.</li>
<li>Secondary side: fixed 20&nbsp;&deg;C thermal boundary with
  UA = 1.333&nbsp;MW/K distributed across <code>nHxNodes</code> cells.</li>
<li>Core heat is deposited in a 5-node fuel-meat half-slab heat structure
  derived from the OpenMC concentric ring defaults.  Heat conducts from the
  fuel centerline symmetry plane to a convective wall coupled to each axial
  coolant node.</li>
</ul>
<p>
<b>FMI inputs:</b> <code>totalPower</code>&nbsp;[W],
<code>axialPowerFractions[n]</code><br/>
<b>FMI outputs:</b> <code>T_inlet</code>&nbsp;[K],
<code>T_outlet</code>&nbsp;[K], <code>massFlow</code>&nbsp;[kg/s],
<code>dp_core</code>&nbsp;[Pa],
<code>T_fuelCenterlineMax</code>&nbsp;[K],
<code>T_fuelWallMax</code>&nbsp;[K],
<code>T_fuelEff</code>&nbsp;[K],
<code>T_moderatorEff</code>&nbsp;[K],
<code>rho_m_eff_SI</code>&nbsp;[kg/m3],
<code>rho_m_eff</code>&nbsp;[g/cm3]
</p>
</html>"),
    experiment(StopTime = 6000, Interval = 5.0, Tolerance = 1e-6),
    uses(Modelica(version = "4.1.0")));

end ResearchReactorThermalHydraulics;
