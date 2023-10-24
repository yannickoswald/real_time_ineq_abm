# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:25:36 2023

@author: earyo
"""


import os
import pandas as pd
import sys
import numpy as np

#%% 
''' MINI EXAMPLE PREP'''
#3 agents one much richer than the other two, three ensemble members
agent_data = np.array([[150, 170, 130], [10, 15, 5], [5, 5, 5]]) 

# 2 observation variables, rich and poor agents as a macro state
obs_ensemble = np.array([[110, 185, 150], [10, 11, 9]])
## current observation and current observation variance
current_obs = [150,11]
current_obs_var = np.var(obs_ensemble, axis = 1) ### for this example we simply set the current obs variance in accodrance with the observation ensemble
### adjust covariance in case of experiment
''' here adjust cov '''
## mini H operator summarizes three agents to two wealth classes 
H = np.array([[1, 0, 0], [0, 0.5, 0.5]])

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
    Kalman_Gain = C @ H.T @ np.linalg.inv(diff) ### how we know it is invertible?
    return Kalman_Gain

def state_update(Kalman_Gain, agent_data, data_ensemble, H):
   
    diff = data_ensemble - H @ agent_data
    X = agent_data + Kalman_Gain @ diff ## micro states
    Y = H @ X ## obsveration corresponded macro states
    
    return diff, X, Y



# must apply current_obs_variance
data_covariance = make_data_covariance(current_obs_var)
Kalman_Gain = make_gain_matrix(agent_data, data_covariance , H)
output = state_update(Kalman_Gain, agent_data, obs_ensemble, H)


#%%




'''  
BIGGER EXAMPLE NOW NOT DONE ANYMORE 
path = "."
### LOAD empirical monthly wealth Data
with open(os.path.join(path, 'agent_example.csv')) as f0:
    d1 = pd.read_csv(f0, encoding = 'unicode_escape')
    
with open(os.path.join(path, 'data_example.csv')) as f1:
    d2 = pd.read_csv(f1, encoding = 'unicode_escape')

with open(os.path.join(path, 'H_example.csv')) as f2:
    d3 = pd.read_csv(f2, encoding = 'unicode_escape', index_col = None, header = None)
    
with open(os.path.join(path, 'current_obs_example.csv')) as f3:
    d4 = pd.read_csv(f3, encoding = 'unicode_escape', index_col = None, header = None)

with open(os.path.join(path, 'current_obs_var_example.csv')) as f4:
    d5 = pd.read_csv(f4, encoding = 'unicode_escape', index_col = None, header = None)
          
    
#### subset data right before the kalman update state in the current implementation
agent_data = np.array(d1.iloc[50:60,:]) ## 10 agents over 5 simulations
obs_ensemble = np.array(d2.iloc[12:15,:]) ## 10 agents over 5 simulations
H = np.array(d3)
'''