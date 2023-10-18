# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:24:46 2023

@author: earyo
"""

import os
import pandas as pd
import sys
import numpy as np
import random
from tqdm import tqdm  ### package for progress bars
import matplotlib.pyplot as plt
from model2_class import Model2
from inequality_metrics import find_wealth_groups
from enkf_yo2 import EnsembleKalmanFilter2



def prepare_enkf2():

    #%%
    path = ".."
    ### LOAD empirical monthly wealth Data
    with open(os.path.join(path, 'data', 'wealth_data_for_import.csv')) as f:
        d1 = pd.read_csv(f, encoding = 'unicode_escape')

    ### LOAD empirical monthly wealth Data sorted by group for state vector check
    with open(os.path.join(path, 'data', 'wealth_data_for_import2.csv')) as f2:
        d2 = pd.read_csv(f2, encoding = 'unicode_escape')
    #%%

    ### let us say the state vector is the share of wealth
    ### of 4 wealth groups top 1%, top 10% etc.
    num_agents = 100
    filter_params = {"ensemble_size": 10,
                     "macro_state_vector_length": 4,
                     "micro_state_vector_length": num_agents}

    model_params = {"population_size": 100, 
                    "concavity": 1,
                    "growth_rate": 0.025, 
                    "start_year": 1990,
                    "adaptive_sensitivity": 0.02}
    
    enkf = EnsembleKalmanFilter2(Model2, filter_params, model_params)
    print("EnKF micro state ensemble:\n", enkf.micro_state_ensemble)
    print("EnKF macro state ensemble:\n", enkf.macro_state_ensemble)

    return enkf


def run_enkf2(enkf):

    time_horizon = 29*12 ## 29 years * 12 months
    for i in tqdm(range(time_horizon), desc="Iterations"):
        #if i == 1: break
        ### set update to false or true
        if i % 100 != 0 or i == 0:
            enkf.step(update = False)
            test = enkf.plot_macro_state(False)
        else:
            enkf.step(update = False)
            
           

def plot_enkf2(enkf):
    #enkf.plot_micro_state()
    #enkf.plot_macro_state(log_var = True)
    return enkf.plot_error(), enkf.plot_fanchart()



if __name__=="__main__":
    enkf = prepare_enkf2()
    run_enkf2(enkf)
    plot_enkf2(enkf)
    print("EnKF micro state ensemble:\n", enkf.micro_state_ensemble)
    print("EnKF macro state ensemble:\n", enkf.macro_state_ensemble)


