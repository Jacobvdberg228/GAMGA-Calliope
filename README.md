# GAMGA: Genetic Algorithm for Modeling to Generate Alternatives

GAMGA is an optimization algorithm designed for energy systems that combines a genetic algorithm with a modeling to generate alternatives (MGA) method. The algorithm and the small model developed have different technologies installed at different locations which can vben cost optimized. The algorithm will search for alternative feasible solutions in the solution space with the solution space being determined by the cost optimal value and the introduced slack.  

## Installation

Calliope is used to design the energy system model. The current Calliope version that is used for this model is 0.6.10. Installing Calliope can be done via this link: https://calliope.readthedocs.io/en/stable/user/installation.html

One important package that is used during the algorithm is DEAP. DEAP is a novel evolutionary computation framework for rapid prototyping and testing of ideas. More information about DEAP can be found here: https://deap.readthedocs.io/en/master/

Gurobi is used as the main solver for this model, but other solvers can also be used. Check the solver guide via the Calliope guide on how to install different solvers such as CBC or GLPK (https://calliope.readthedocs.io/en/stable/user/installation.html#solvers).

## Usage

All the model files are placed in the GAMGA_model folder. model_(res).yaml files represent the main format of the model that will be run. If you use a different solver for example, different settings need to be specified in this file. Again, more about this can be found when looking at the CAlliope link. 

To run the model you have the model_run.ipynb file. Make sure that when initializing the model you refer to the correct path where your model.yaml files are located. Variables such as generation number, population size, population number, mutation rate and crossover rate can be specified in the variable tab. The algorithm should be easy applicable to other Calliope models. 

Take notice that technologies such as demand and transmission are excluded from making up the individual. 

Some small adjustments that are specific to the model must be made.
- Initialization values (-0.01, 0.01) for first generation
- Max cap value, which defines the max cap of a technology if its set at inf
- elite rate

![License](https://img.shields.io/badge/license-MIT-blue)
