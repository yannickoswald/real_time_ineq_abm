# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:24:53 2023

@author: earyo
"""
#General packages
import os
import numpy as np
from tqdm import tqdm  ### package for progress bars
os.chdir(".")
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

#%%   PLOT FIGURE 2 showing typical model run of both models 
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4))
economy1.plot_wealth_groups_over_time(ax1, time_horizon)
economy2.plot_wealth_groups_over_time(ax2)
ax1.set_title("Model 1")
ax2.set_title("Model 2 (Network-based ABM)")
plt.savefig('fig2.png',  bbox_inches='tight', dpi=300)


#%%   PLOT FIGURE 2 showing mean of n typical model run of both models 
'''
n = 10 ## run models n times and write out data 

for i in range(n):
    ## define time horizon
    time_horizon = 29*12 ## 29 years * 12 months | from Jan 1990 to Dec 2018
    ### initialize model 1
    economy1.make_agents()
    ### run the models
    for i in tqdm(range(time_horizon)):
        economy1.step()
        economy2.step()

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4))
economy1.plot_wealth_groups_over_time(ax1, time_horizon)
economy2.plot_wealth_groups_over_time(ax2)
ax1.set_title("Model 1")
ax2.set_title("Model 2 (Network-based ABM)")
plt.savefig('fig2.png',  bbox_inches='tight', dpi=300)
'''
