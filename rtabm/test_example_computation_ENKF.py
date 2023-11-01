# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:25:36 2023

@author: earyo
"""

#%% 

import random
import numpy as np

class KalmanToy():

    def __init__(self, mean, std_dev, rich_agents=None, rich_agents_obs=None):
        self.mean = mean
        self.std_dev = std_dev
        self.agent_data = None  # specified through create_agent_state
        self.obs_ensemble = None  # specified through create_agent_state
        self.current_obs = [150, 11]
        self.current_obs_var = None  # to be set later
        self.H = np.array([[1, 0, 0], [0, 0.5, 0.5]])
        self.create_agent_states(rich_agents)  # Populate self.agent_data
        self.create_obs_ensemble(rich_agents_obs)  # Populate self.obs_ensemble
        self.current_obs_var = np.var(self.obs_ensemble, axis=1)  # Now we can set this

    def create_agent_states(self, rich_agents=None):
        if rich_agents is None:
            rich_agents = []
            for i in range(3):
                random_number = random.gauss(self.mean, self.std_dev)
                clipped_number = min(200, max(100, random_number))    
                rich_agents.append(clipped_number)  
        self.agent_data = np.array([rich_agents, [10, 15, 5], [5, 5, 5]])

    def create_obs_ensemble(self, rich_agents_obs=None): 
        if rich_agents_obs is None:
            rich_agents_obs = []
            for i in range(3):
                random_number = random.gauss(self.mean, self.std_dev)
                clipped_number = min(200, max(100, random_number))    
                rich_agents_obs.append(clipped_number) 
        self.obs_ensemble = np.array([rich_agents_obs, [10, 11, 9]])
        
    def make_data_covariance(self, data):
        return np.diag(data)

    def make_gain_matrix(self):
        C = np.cov(self.agent_data)
        self.current_obs_var = np.var(self.obs_ensemble, axis=1)
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
        return diff, X, Y, Kalman_Gain @ diff

# Usage without specifying rich_agents and rich_agents_obs
kalman_toy1 = KalmanToy(150, 20)
print("Without custom rich_agents and rich_agents_obs")
print("Agent Data:", kalman_toy1.agent_data)
print("Observation Ensemble:", kalman_toy1.obs_ensemble)

# Usage with specifying rich_agents and rich_agents_obs
kalman_toy2 = KalmanToy(150, 20, [150, 170, 130], [110, 185, 150])
data2 = kalman_toy2.state_update()
print("\nWith custom rich_agents and rich_agents_obs")
print("Agent Data:", kalman_toy2.agent_data)
print("Observation Ensemble:", kalman_toy2.obs_ensemble)
