# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:47:43 2023

@author: earyo
"""
# Imports
import warnings as warns
import numpy as np
import pandas as pd

# Classes
class EnsembleKalmanFilter:
    """
    A class to represent a EnKF for application with a wealth 
    agent-based model for the United States.
    """
    def __init__(self, model, filter_params, model_params):
        """
        Initialise the Ensemble Kalman Filter.
        Params:
            model
            filter_params
            model_params
        Returns:
            None
        """
        
        self.ensemble_size = None
        self.state_vector_length = None
        
        # Get filter attributes from params, warn if unexpected attribute
        for k, v in filter_params.items():
             if not hasattr(self, k):
                 w = 'EnKF received unexpected {0} attribute.'.format(k) 
                 warns.warn(w, RuntimeWarning)
             setattr(self, k, v)
        
        #print(model, model_params)    
        # Set up ensemble of models and other global properties
        self.models = [model(**model_params) for _ in range(self.ensemble_size)]
        shape_ens = (self.state_vector_length, self.ensemble_size)
        self.state_ensemble = np.zeros(shape=shape_ens)
        self.ensemble_covariance = None
        self.data_ensemble = None 
        self.data_covariance = None
        self.Kalman_Gain = None
        self.state_mean = None
        self.time = 0 
        
        
        ### load observation data
        ### LOAD empirical monthly wealth Data sorted by group
        ### for state vector check
        with open('./data/wealth_data_for_import2.csv') as f2:
            self.data = pd.read_csv(f2, encoding = 'unicode_escape')
            
            
        y = model_params["start_year"]
        idx_begin = min((self.data[self.data["year"]==1990].index.values))
        
        self.obs = self.data.iloc[idx_begin::][["year","month",
                                    "real_wealth_share",
                                    "variance_real_wealth_share"]]
    
    def predict(self):
        """
        Step the model forward by one time-step to produce a prediction.
        Params:
        Returns:
            None
        """
        for i in range(self.ensemble_size):
            self.models[i].step()
        self.time = self.models[0].time 
        
    def set_current_obs(self):
        """
        Here we set the current observation corresponding to the time
        in the models as well as the variance of the current observation.
        This is used in the method update_data_ensemble"""
        
        self.current_obs = self.obs.iloc[self.time*4-4:self.time*4, 2]
        self.current_obs_var = self.obs.iloc[self.time*4-4:self.time*4, 3]
        
    def update_state_ensemble(self):
        """
        Update self.state_ensemble based on the states of the models.
        """
        for i in range(self.ensemble_size):
            self.state_ensemble[:, i] = self.models[i].macro_state
            
    def update_state_mean(self):
            """
            Update self.state_mean based on the current state ensemble.
            """
            self.state_mean = np.mean(self.state_ensemble, axis=1)
        
    def make_ensemble_covariance(self):
        """
        Create ensemble covariance matrix.
        """
        self.ensemble_covariance = np.cov(self.state_ensemble)
      
    def make_data_covariance(self):
        """
        Create data covariance matrix.
        """
        self.data_covariance = np.diag(self.current_obs_var)

    def update_data_ensemble(self):
        """
        Create perturbed data vector.
        I.e. a replicate of the data vector plus normal random n-d vector.
        R - data (co?)variance; this should be either a number or a vector with
        same length as the data.
        """
        x = np.zeros(shape=(len(self.current_obs), self.ensemble_size)) 
        for i in range(self.ensemble_size):
            err = np.random.normal(0, self.current_obs_var, len(self.current_obs))
            x[:, i] = self.current_obs + err
        self.data_ensemble = x
        
    def make_gain_matrix(self):
        """
        Create kalman gain matrix.
        """
        diff = self.ensemble_covariance + self.data_covariance
        self.Kalman_Gain = self.ensemble_covariance @ np.linalg.inv(diff) 
        
    def state_update(self):
        """
        Update system state of model. This is the state update equation of the 
        Kalman Filter.
        """
        X = np.zeros(shape=(self.state_vector_length, self.ensemble_size))
        for i in range(self.ensemble_size):
            diff = self.data_ensemble[:, i] - self.state_ensemble[:, i]
            X[:, i] = self.state_ensemble[:, i] + self.Kalman_Gain @ diff
        self.state_ensemble = X

