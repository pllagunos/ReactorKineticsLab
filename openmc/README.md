# OpenMC modeling of reactor core

This folder explores the modeling of a more realistic, monte-carlo  simulated, nuclear reactor core instead of the naive homogeneous reactor used in `theory/reactorModel.ipynb`

## FRM2 nat
Involuted plate reactor loosely based in the FRM2 reactor in Garching, Munich. Uses natural uranium and heavy water mdoerator. Dimentions for critical core were ballparked via `optimization.py` but ultimately it was physical intuition and iteration to arrive at a "good enough" core geometry. Achieved k_eff \approx 1 but ultimately prefered to model an annular concentric core because of the simpler computational costs

## Concentric model
Concentric annuli rings of natural uranium moderated by heavy water. Loosely based in the Misouri Research Reactor. Again the critical geometry was found by iteration and intuition. `concentric_modeling.ipynb` gives important parameters such as fuel fraction and fuel mass when constructing a model, which serves as the physical compass during modeling. 

# Architecture
- `build` folder where xml files and statepoint files as well as multigroup cross sections are saved.
- `concentric_modeling.ipynb` notebook where the overall scope and modeling of the concentric core is done
- `frm2_nat.ipynb` notebook where the overalls cope and modeling of the involute natural uranium core is done
- `exportMGXS.ipynb` notebook where multi-group cross sections used in deterministic codes are generated based on existing models in the build folder.
- `involutes.py` helper functions and classes to build involute plates
- `fuel_element.py` builds openmc fuel element with involute plates
- `concentric_fuel` build openmc concentric annuli fuel element
- `reactor_geometry.py` builds full geometry of reactor including fuel, moderator and reflector tanks in openmc.
- `ploting.py` helper functions to plot openmc universes and meshes with flux, fission and entropy tallies
- `build_simulation.py` modular file which builds the openmc eigenvalue simulation model for either the involute or concentric fuel core.
- `mgxs_export.py` builds multigroup tallies and reactor model, loads MGXS and writes them to json and csv files. Validates constants by running a multigroup openmc simulation. Workhorse used with exportMGXS.ipynb
