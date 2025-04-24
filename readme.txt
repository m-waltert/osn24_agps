The following files have been used for the paper "Evaluating Potential Fuel Savings of External Alternative Ground Propulsion Systems"
presented at the 12th OpenSky Network Symposium in Hamburg, October 8, 2024 as well as the ATRS World Conference 2025 contribution.

# Files used for OSN24:
* agps_proc.ipynb: Download OSN data, preprocessing, surface event detection, estimation of taxi fuel consumption
* agps_plot.ipynb: Use this file to create Figures 3 to 5 as well as all Tables shown in the paper
* pushbackplot.ipynb: Use this file to create Figure 1 shown in the paper.
* agps_funs.py: Contains all functions used
* agps_config.py: Contains configuration/assumptions used for the paper

# Files used for ATRS25:
* agps_atrs25.ipynb: Reads df_movements (output of OSN24), solves optimization problem for AGPS units assignment to aircraft, and plots results
* apgs_optim.py: Contains functions (e.g. the optimization problem), used in agps_atrs25.ipynb


Manuel Waltert, April 24, 2025.