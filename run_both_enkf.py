# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:24:53 2023

@author: earyo
"""


### MODEL 1 infrastructure
import os
import pandas as pd
import sys
import numpy as np
import random
from tqdm import tqdm  ### package for progress bars
os.chdir(".")
import matplotlib.pyplot as plt
from economy_class_wealth import Economy
from inequality_metrics import find_wealth_groups
from enkf_yo import EnsembleKalmanFilter

### LOAD empirical monthly wealth Data
with open('./data/wealth_data_for_import.csv') as f:
    d1 = pd.read_csv(f, encoding = 'unicode_escape')
    
### LOAD empirical monthly wealth Data sorted by group for state vector check
with open('./data/wealth_data_for_import2.csv') as f2:
    d2 = pd.read_csv(f2, encoding = 'unicode_escape')

### MODEL 2

from network_abm import *
import pandas as pd
from enkf_yo2 import EnsembleKalmanFilter




#%% run both base models for fit to the data plot in calibration section
time_horizon = 29*12 ## 29 years * 12 months

model = Model(500,
              concavity=1,
              growth_rate = 0.02, 
              start_year = 1990,
              adaptive_sensitivity=0.02)  # 100 agents
for _ in tqdm(range(time_horizon)):  # Run for 10 steps
    model.step()
model.plot_network()
model.plot_wealth_histogram()

#%%

# NM Naming the parameters isn't necessary but makes it easier for me to understand
economy = Economy(population_size=500, growth_rate=0.025, b_begin=1.3, distribution="Pareto_lognormal", start_year=1990)
### one-time procedure
economy.make_agents()
list_agents = economy.agents
array_agent_wealth = np.asarray([x.wealth for x in economy.agents])
for i in tqdm(range(time_horizon)):
    economy.step()



#%%    
fig, (ax1, ax2) = plt.subplots(1, 2)
economy.plot_wealth_groups_over_time(ax1, time_horizon)
model.plot_wealth_groups_over_time(ax2)

#%%
'''
#%%

### let us say the state vector is the share of wealth 
### of 4 wealth groups top 1%, top 10% etc.
num_agents = 100
filter_params = {"ensemble_size": 10,
                 "macro_state_vector_length": 4,
                 "micro_state_vector_length": num_agents}

model_params = {"population_size": num_agents,
 "growth_rate": 0.025,
 "b_begin": 1.3,
 "distribution": "Pareto_lognormal",
 "start_year": 1990 }


enkf = EnsembleKalmanFilter(Economy, filter_params, model_params)
print(enkf.micro_state_ensemble)
print(enkf.macro_state_ensemble)
time_horizon = 12*29 ## 29 years * 12 months
for i in tqdm(range(time_horizon)):
    #if i == 1: break 
    ### set update to false or true
    if i % 100 != 0 or i == 0: 
        enkf.step(update = False)
        test = enkf.plot_macro_state(False)
    else:
        enkf.step(update = True)
        

    
enkf.plot_fanchart()
enkf.plot_micro_state()
enkf.plot_macro_state(log_var = True)
print(enkf.micro_state_ensemble)
print(enkf.macro_state_ensemble)'''