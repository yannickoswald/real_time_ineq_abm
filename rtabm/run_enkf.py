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


def prepare_enkf():

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
    filter_params = {"ensemble_size": 20,
                     "macro_state_vector_length": 4,
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

    time_horizon = 12*29 ## 29 years * 12 months
    for i in tqdm(range(time_horizon), desc="Iterations"):
        #if i == 1: break
        ### set update to false or true
        if i % 100 != 0 or i == 0:
            enkf.step(update = False)
            test = enkf.plot_macro_state(False)
        else:
            enkf.step(update = True)

def plot_enkf(enkf):
    enkf.plot_fanchart()
    enkf.plot_micro_state()
    enkf.plot_macro_state(log_var = True)

if __name__=="__main__":
    model = prepare_enkf()
    run_enkf(model)
    plot_enkf(model)
    print("EnKF micro state ensemble:\n", model.micro_state_ensemble)
    print("EnKF macro state ensemble:\n", model.macro_state_ensemble)



        