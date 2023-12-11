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


def prepare_enkf(Model, model_params, ensemble_size, macro_state_dim, filter_freq, uncertainty_obs):
    
    assert macro_state_dim == 3 or macro_state_dim == 4 , "Incorrect dimensions for macro state."

    ### LOAD empirical monthly wealth Data
    with open(os.path.join("..", 'data', 'wealth_data_for_import.csv')) as f:
        d1 = pd.read_csv(f, encoding = 'unicode_escape')

    ### LOAD empirical wealth Data sorted by group for state vector check
    with open(os.path.join("..", 'data', 'wealth_data_for_import2.csv')) as f2:
        d2 = pd.read_csv(f2, encoding = 'unicode_escape')


    ### let us say the state vector is the share of wealth
    ### of 4 wealth groups top 1%, top 10% etc.
    filter_params = {"ensemble_size": ensemble_size,
                     "macro_state_vector_length": macro_state_dim,
                     "micro_state_vector_length": model_params["population_size"]}

    enkf = EnsembleKalmanFilter(Model, filter_params, model_params, constant_a = uncertainty_obs, filter_freq = filter_freq)
    #print("EnKF micro state ensemble:\n", enkf.micro_state_ensemble)
    #print("EnKF macro state ensemble:\n", enkf.macro_state_ensemble)

    return enkf

def run_enkf(enkf, time_horizon, filter_freq):
    
    # Set a default value for filter_freq if not provided
    if filter_freq is None:
        filter_freq = 30  # default value, can be adjusted as needed

    # Ensure filter_freq is not zero to avoid division by zero error
    if filter_freq == 0:
        raise ValueError("filter_freq cannot be zero.")

    #time_horizon = 29*12 ## 29 years * 12 months
    for i in tqdm(range(time_horizon), desc="Iterations ENKF Model 1"):
    #for i in range(time_horizon):    
        #if i == 1: break
        ### set update to false or true
        if i % filter_freq != 0 or i == 0:
            enkf.step(update = False)
            #test = enkf.plot_macro_state(False)
        else:
            enkf.step(update = True)

def plot_enkf(enkf):
    #enkf.plot_micro_state()
    #enkf.plot_macro_state(log_var = True)
    fig, ax = plt.subplots(figsize = (8,6))
    enkf.plot_fanchart(ax) ### now an axis needs to be passed to this plot 
    enkf.plot_error()
