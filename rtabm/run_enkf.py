# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:58:26 2023
@author: earyo
"""
import os
import pandas as pd
import sys
import numpy as np
import random
from tqdm import tqdm  ### package for progress bars
import matplotlib.pyplot as plt
from model1_class import Model1
from inequality_metrics import find_wealth_groups
from enkf_yo import EnsembleKalmanFilter


def prepare_enkf(num_agents:int, ensemble_size:int, macro_state_dim: int):
    
    assert macro_state_dim == 3 or macro_state_dim == 4 , "Incorrect dimensions for macro state."

    path = ".."
    ### LOAD empirical monthly wealth Data
    with open(os.path.join(path, 'data', 'wealth_data_for_import.csv')) as f:
        d1 = pd.read_csv(f, encoding = 'unicode_escape')

    ### LOAD empirical monthly wealth Data sorted by group for state vector check
    with open(os.path.join(path, 'data', 'wealth_data_for_import2.csv')) as f2:
        d2 = pd.read_csv(f2, encoding = 'unicode_escape')


    ### let us say the state vector is the share of wealth
    ### of 4 wealth groups top 1%, top 10% etc.
    filter_params = {"ensemble_size": ensemble_size,
                     "macro_state_vector_length": macro_state_dim,
                     "micro_state_vector_length": num_agents}

    model_params = {"population_size": num_agents,
     "growth_rate": 0.025,
     "b_begin": 1.3,
     "distribution": "Pareto_lognormal",
     "start_year": 1990 }


    enkf = EnsembleKalmanFilter(Model1, filter_params, model_params)
    print("EnKF micro state ensemble:\n", enkf.micro_state_ensemble)
    print("EnKF macro state ensemble:\n", enkf.macro_state_ensemble)

    return enkf

def run_enkf(enkf):

    time_horizon = 10 ## 29 years * 12 months
    for i in tqdm(range(time_horizon), desc="Iterations"):
        #if i == 1: break
        ### set update to false or true
        if i % 5 != 0 or i == 0:
            enkf.step(update = False)
            test = enkf.plot_macro_state(False)
        else:
            enkf.step(update = True)

def plot_enkf(enkf):
    #enkf.plot_micro_state()
    #enkf.plot_macro_state(log_var = True)
    fig, ax = plt.subplots(figsize = (8,6))
    enkf.plot_fanchart(ax) ### now an axis needs to be passed to this plot 
    enkf.plot_error()

if __name__=="__main__":
    ### if num_agents >= 100 then macro_state_dim = 4, otherwise = 3
    enkf1 = prepare_enkf(num_agents=10, ensemble_size= 5, macro_state_dim = 3)
    run_enkf(enkf1)
    plot_enkf(enkf1)


### save dataframe with all data necessary for 5 x 10 example (5 ensemble simulations, 10 agents)
H = enkf1.H ## general obs. operator
## all agent data 
agents_data = enkf1.micro_history
### all data/observation things
data_ensemble = enkf1.data_ensemble_history
current_obs_hist = enkf1.current_obs_history
current_obs_var_hist = enkf1.current_obs_var_history




columns=['Column1', 'Column2', 'Column3',
         'Column4', 'Column5'] 


# Vertically stack arrays into a 2D array
stacked_array = np.vstack(agents_data)
stacked_array2 = np.vstack(data_ensemble)
stacked_array3 = np.vstack(current_obs_hist)
stacked_array4 = np.vstack(current_obs_var_hist)
# Convert list of arrays to DataFrame
df1 = pd.DataFrame(stacked_array, columns = columns)
df2 = pd.DataFrame(stacked_array2, columns = columns)
df3 = pd.DataFrame(stacked_array3)
df4 = pd.DataFrame(stacked_array4)

# Save the DataFrame to a CSV file
df1.to_csv('agent_example.csv', index=False)
df2.to_csv('data_example.csv', index=False)
df3.to_csv('current_obs_example.csv', index=False)
df4.to_csv('current_obs_var_example.csv', index=False)
# Save array to CSV file
np.savetxt('H_example.csv', H, delimiter=',')

