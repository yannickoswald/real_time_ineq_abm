# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:25:36 2023

@author: earyo
"""


import os
import pandas as pd
import sys
import numpy as np
import random
#%% 



import random
import numpy as np

class KalmanToy():

    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev
        self.agent_data = self.create_agent_states()
        self.obs_ensemble = self.create_obs_ensemble()
        self.current_obs = [150, 11]
        self.current_obs_var = np.var(self.obs_ensemble, axis=1)
        self.H = np.array([[1, 0, 0], [0, 0.5, 0.5]])
    
    def create_agent_states(self):
        rich_agents = list()
        for i in range(3):
            random_number = random.gauss(self.mean, self.std_dev)
            clipped_number = min(200, max(100, random_number))    
            rich_agents.append(clipped_number)  
        agents = np.array([rich_agents, [10, 15, 5], [5, 5, 5]]) 
        return agents 

    def create_obs_ensemble(self): 
        rich_agents_obs = list()
        for i in range(3):
            random_number = random.gauss(self.mean, self.std_dev)
            clipped_number = min(200, max(100, random_number))    
            rich_agents_obs.append(clipped_number) 
        obs_ensemble = np.array([rich_agents_obs, [10, 11, 9]])
        return obs_ensemble

    def make_data_covariance(self, data):
        return np.diag(data)

    def make_gain_matrix(self):
        C = np.cov(self.agent_data)
        data_covariance = self.make_data_covariance(self.current_obs_var)
        state_covariance = self.H @ C @ self.H.T
        diff = state_covariance + data_covariance
        Kalman_Gain = C @ self.H.T @ np.linalg.inv(diff)
        return Kalman_Gain, state_covariance, C

    def state_update(self):
        Kalman_Gain, state_covariance, C = self.make_gain_matrix()
        diff = self.obs_ensemble - self.H @ self.agent_data
        X = self.agent_data + Kalman_Gain @ diff
        Y = self.H @ X
        return diff, X, Y


# Usage
kalman_toy = KalmanToy(150, 20)
Kalman_Gain, state_covariance, C = kalman_toy.make_gain_matrix()
diff, X, Y = kalman_toy.state_update()
agent_states = kalman_toy.agent_data
obs_ensemble = kalman_toy.obs_ensemble
print("diff:", diff)
print("X:", X)
print("Y:", Y)

