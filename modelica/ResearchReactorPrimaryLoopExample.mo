model ResearchReactorPrimaryLoopExample
  "Standalone driver for ResearchReactorPrimaryLoop"
  extends Modelica.Icons.Example;

  parameter Integer nAxialNodes = 8;

  ResearchReactorPrimaryLoop primaryLoop(
    nAxialNodes = nAxialNodes,
    defaultValveOpening = 0.5)
    annotation(Placement(transformation(extent = {{-20, -20}, {20, 20}})));

  Modelica.Blocks.Sources.Constant totalPower(k = 20e6)
    annotation(Placement(transformation(extent = {{-80, 20}, {-60, 40}})));

  Modelica.Blocks.Sources.Constant valveOpening(k = 1.0)
    annotation(Placement(transformation(extent = {{-80, -20}, {-60, 0}})));

  Modelica.Blocks.Sources.Constant axialProfile[nAxialNodes](
    each k = 1.0 / nAxialNodes)
    annotation(Placement(transformation(extent = {{-80, -60}, {-60, -40}})));

equation
  connect(totalPower.y, primaryLoop.totalPower)
    annotation(Line(points = {{-59, 30}, {-30, 30}, {-30, -2}, {-20, -2}}, color = {0, 0, 127}));
  connect(valveOpening.y, primaryLoop.valveOpening)
    annotation(Line(points = {{-59, -10}, {-30, -10}, {-30, -10}, {-20, -10}}, color = {0, 0, 127}));

  for i in 1:nAxialNodes loop
    connect(axialProfile[i].y, primaryLoop.axialPowerFractions[i]);
  end for;

  annotation(
    Documentation(info = "<html>
<p>
This example binds the external inputs of <code>ResearchReactorPrimaryLoop</code>
to simple constants so the model can be translated and simulated without an FMU
master or co-simulation driver.
</p>
</html>"),
    experiment(StopTime = 100));
end ResearchReactorPrimaryLoopExample;
