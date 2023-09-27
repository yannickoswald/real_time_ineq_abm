# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:24:53 2023

@author: earyo
"""
#General packages
import os
import numpy as np
from tqdm import tqdm  ### package for progress bars
os.chdir("..")
import matplotlib.pyplot as plt
### MODEL 1 infrastructure
from model1_class import Model1
from enkf_yo import EnsembleKalmanFilter
### MODEL 2 infrastructure
from model2_class import Model2
from enkf_yo2 import EnsembleKalmanFilter

#%%

# Set up both model economies
economy1 = Model1(population_size=500,
                  growth_rate=0.025, 
                  b_begin=1.3,
                  distribution="Pareto_lognormal",
                  start_year=1990)

economy2 = Model2(500,
              concavity=1,
              growth_rate = 0.02, 
              start_year = 1990,
              adaptive_sensitivity=0.02)

## define time horizon
time_horizon = 29*12 ## 29 years * 12 months | from Jan 1990 to Dec 2018
### initialize model 1
economy1.make_agents()
### run the models
for i in tqdm(range(time_horizon)):
    economy1.step()
    economy2.step()

#%%   PLOT FIGURE 3 showing typical model run of both models 
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4))
economy1.plot_wealth_groups_over_time(ax1, time_horizon)
economy2.plot_wealth_groups_over_time(ax2)
ax1.set_title("Model 1")
ax2.set_title("Model 2 (Network-based ABM)")
plt.savefig('fig2.png',  bbox_inches='tight', dpi=300)

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