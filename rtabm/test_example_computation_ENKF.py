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
    

agent_data = np.array(d1.iloc[50:60,:]) ## 10 agents over 5 simulations
obs_ensemble = np.array(d2.iloc[12:15,:]) ## 10 agents over 5 simulations

#%%
#prepare enkf computation to recreate exactly one full ENKF step


#### MAKE Kalman Filter

        
def make_gain_matrix():
    """
    Create kalman gain matrix.
    Should be a (n x 4) matrix since in the state update equation we have
    the n-dim vector (because of n-agents) + the update term which 4 dimensional
    so the Kalman Gain needs to make it (n x 1)
    micro_state_ensemble should be num_agents x ensemble_size 
    """

    C = np.cov(self.micro_state_ensemble)
    state_covariance = self.H @ C @ self.H.T
    diff = state_covariance + self.data_covariance
    self.Kalman_Gain = C @ self.H.T @ np.linalg.inv(diff)

    '''
    Keiran version original
    C = np.cov(self.state_ensemble)
    state_covariance = self.H @ C @ self.H_transpose
    diff = state_covariance + self.data_covariance
    return C @ self.H_transpose @ np.linalg.inv(diff)
    '''