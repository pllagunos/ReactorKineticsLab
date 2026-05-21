model ResearchReactorThermalHydraulicsExample
  "Standalone driver: full 20 MW at uniform axial power profile.
  Run with:  omc run_th.mos   from /home/pablo/modelica/ReactorKineticsLab/modelica/"
  extends Modelica.Icons.Example;

  parameter Integer nAxialNodes = 8;

  ResearchReactorThermalHydraulics plant(
    nAxialNodes = nAxialNodes)
    annotation(Placement(transformation(extent = {{-20, -20}, {20, 20}})));

  Modelica.Blocks.Sources.Constant totalPower(k = 20e6)
    annotation(Placement(transformation(extent = {{-80, 10}, {-60, 30}})));

  Modelica.Blocks.Sources.Constant axialProfile[nAxialNodes](
    each k = 1.0 / nAxialNodes)
    annotation(Placement(transformation(extent = {{-80, -30}, {-60, -10}})));

equation
  connect(totalPower.y, plant.totalPower)
    annotation(Line(points = {{-59, 20}, {-40, 20}, {-40, 4}, {-20, 4}},
      color = {0, 0, 127}));

  for i in 1:nAxialNodes loop
    connect(axialProfile[i].y, plant.axialPowerFractions[i]);
  end for;

  annotation(
    experiment(StopTime = 6000, Interval = 5.0, Tolerance = 1e-6),
    uses(Modelica(version = "4.1.0")));

end ResearchReactorThermalHydraulicsExample;
