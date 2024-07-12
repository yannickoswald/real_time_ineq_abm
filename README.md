# "Real-time" inequality agent-based model supported by Ensemble Kalman Filter

## Repository Overview

This repository stores all code related to the paper **An agent-based model of wealth inequality in the US supported with an Ensemble Kalman Filter.** (in preparation/review).

All relevant output figures for this paper are produced in Jupyter notebooks. These notebooks employ the classes that represent the actual model, which are Python files.

### Model Python Files / Classes

1. **model1_class.py** - Defines model #1
2. **model2_class.py** - Defines model #2
3. **agent1_class** - Defines agent type 1 belonging to model #1
4. **agent2_class** - Defines agent type 2 belonging to model #2
5. **enkf_yo** - Defines the Ensemble Kalman Filter (ENKF) used in combination with both models
6. **inequality_metrics** - Bundle of functions to compute the inequality measures e.g. the wealth groups aggregated from the agent-based model
7. **exponential_pareto_avg_distr** - Functions to compute the distributional model to calibrate the agents

### Notebooks for running experiments and figure outputs

1. **calibration_weighted_avg.ipynb** - Calibration of agent initial distribution to empirical wealth distribution
2. **run_both_models_n_times_and_compute_error_jupyter.ipynb** - Ensemble model runs without filter necessary for Figure 2 in the paper
3. **experiment1_jupyter.ipynb**
4. **experiment2_jupyter.ipynb**
5. **experiment3_jupyter.ipynb**
6. **experiment4_jupyter.ipynb**
7. **experiment5_jupyter.ipynb**

### Important Data files

1. **wealth_data_for_import.csv** - contains the wealth inequality data for four wealth groups as taken from https://realtimeinequality.org/
2. **average_wealth_for_every_year** - contains the avg. wealth for every year so the correct scale factor can be chosen, applied to the exponential-pareto distribution, in order to initialise the agents with their wealth

## Getting Started

To set up the environment for this project, you can use the environment.yml file. This file contains all the necessary dependencies and configurations to get you started quickly.

Prerequisites
Ensure you have conda installed on your machine.

## Usage



1. **Clone the repository:**



2. **Navigate to the repository:**


3. **Run the notebooks:**



**Reporting Issues**

If you encounter any issues, please open an issue in the issue tracker. Provide a detailed description of the problem, including steps to reproduce the issue, and any relevant error messages or logs. This will help us address the problem more effectively.

**License**

This project is licensed under the MIT License - see the LICENSE file for details.

**Contact**

y-oswald@web.de
