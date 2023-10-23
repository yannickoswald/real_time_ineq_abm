# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:25:36 2023

@author: earyo
"""


import os
import pandas as pd
import sys
import numpy as np


path = "."
### LOAD empirical monthly wealth Data
with open(os.path.join(path, 'agent_example.csv')) as f0:
    d1 = pd.read_csv(f0, encoding = 'unicode_escape')
    
with open(os.path.join(path, 'data_example.csv')) as f1:
    d2 = pd.read_csv(f1, encoding = 'unicode_escape')

with open(os.path.join(path, 'H_example.csv')) as f2:
    d3 = pd.read_csv(f2, encoding = 'unicode_escape', index_col = None, header = None)
    
#### subset data right before the kalman update state in the current implementation
agent_data = np.array(d1.iloc[50:60,:]) ## 10 agents over 5 simulations
obs_ensemble = np.array(d2.iloc[12:15,:]) ## 10 agents over 5 simulations
H = np.array(d3)
#%%
#prepare enkf computation to recreate exactly one full ENKF step


#### MAKE Kalman Filter



def make_data_covariance(data):
    """
    Create data covariance matrix which assumes no correlation between 
    data time series
    """
    return np.diag(data)

        
def make_gain_matrix(micro_state_ensemble, data_covariance, H):
    """
    Create kalman gain matrix.
    Should be a (n x 3) matrix since in the state update equation we have
    the n-dim vector (because of n-agents) + the update term which is 3 dimensional
    in this test example only because there are only 10 agents
    so the Kalman Gain needs to make it (n x 1)
    micro_state_ensemble should be num_agents x ensemble_size 
    """

    C = np.cov(micro_state_ensemble)
    state_covariance = H @ C @ H.T
    diff = state_covariance + data_covariance
    Kalman_Gain = C @ H.T @ np.linalg.inv(diff)
    return Kalman_Gain



data_covariance = make_data_covariance()
make_gain_matrix(agent_data, data_covariance , H)
