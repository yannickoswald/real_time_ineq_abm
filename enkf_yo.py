# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:47:43 2023

@author: earyo
"""
# Imports
import warnings as warns
import numpy as np
import pandas as pd
from plot_bivariate_distr import *

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
        self.macro_state_vector_length = None
        self.micro_state_vector_length = None
        
        # Get filter attributes from params, warn if unexpected attribute
        for k, v in filter_params.items():
             if not hasattr(self, k):
                 w = 'EnKF received unexpected {0} attribute.'.format(k) 
                 warns.warn(w, RuntimeWarning)
             setattr(self, k, v)
        
        #print(model, model_params)    
        # Set up ensemble of models and other global properties
        self.models = [model(**model_params) for _ in range(self.ensemble_size)]
        shape_macro = (self.macro_state_vector_length, self.ensemble_size)
        shape_micro = (self.micro_state_vector_length, self.ensemble_size)
        
        self.macro_state_ensemble = np.zeros(shape=shape_macro)
        self.micro_state_ensemble = np.zeros(shape=shape_micro)
        #### Observation matrix = translation matrix between macro and micro
        #### states
        self.H = self.make_H(self.micro_state_vector_length, 4).T
        
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
                                    "real_wealth_per_unit",
                                    "variance_real_wealth"]]
    
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
            self.macro_state_ensemble[:, i] = self.models[i].macro_state
            self.micro_state_ensemble[:, i] = self.models[i].micro_state
            
    def update_state_mean(self):
            """
            Update self.state_mean based on the current state ensemble.
            """
            self.state_mean = np.mean(self.macro_state_ensemble, axis=1)
        
    def make_ensemble_covariance(self):
        """
        Create ensemble covariance matrix.
        """
        self.ensemble_covariance = np.cov(self.macro_state_ensemble)
      
    def make_data_covariance(self):
        """
        Create data covariance matrix.
        """
        self.data_covariance = np.diag(self.current_obs_var)

    def make_H(self, dim_micro_state, dim_data):
        H = np.zeros((dim_micro_state, dim_data))
        L = self.micro_state_vector_length
        H[:int(L*0.01),0] = 1
        H[:int(L*0.1),1] = 1
        H[int(L*0.1):int(L*0.5),2] = 1
        H[int(L*0.5):,3] = 1
        return H
    
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
        Should be a (n x 4) matrix since in the state update equation we have
        the n-dim vector (because of n-agents) + the update term which 4 dimensional
        so the Kalman Gain needs to make it (n x 1)
        """
        C = np.cov(self.micro_state_ensemble)
        state_covariance = self.H @ C @ self.H.T
        diff = state_covariance + self.data_covariance
        self.Kalman_Gain = C @ self.H.T @ np.linalg.inv(diff)
        
    def state_update(self):
        """
        Update system state of model. This is the state update equation of the 
        Kalman Filter. Here the deceive step then is that the difference is 
        calculated between self.data_ensemble (4-dimensional) and self.micro_state_ensemble
        (n-dimensional) so there the observation matrix H
        does a translation between micro and macro state. And 
        we update the micro-level state vector which is n-
        dimensional based on n-agents.
        
        """
        X = np.zeros(shape=(self.micro_state_vector_length, self.ensemble_size))
        Y = np.zeros(shape=(self.macro_state_vector_length, self.ensemble_size))
        for i in range(self.ensemble_size):
            diff = self.data_ensemble[:, i] - self.H @ self.micro_state_ensemble[:, i]
            print('this is the diff ', diff)
            X[:, i] = self.micro_state_ensemble[:, i] + self.Kalman_Gain @ diff
            Y[:, i] =  self.H @ X[:, i]
        self.micro_state_ensemble = X
        self.macro_state_ensemble = Y
        
    def update_models(self):
        """
        Update individual model states based on state ensemble.
        """
        #for i in range(self.ensemble_size):
         #   self.models[i].state = self.state_ensemble[:, i]
        pass ### will be filled later

    def plot_macro_state(self):
        x = self.macro_state_ensemble[0,:]
        y = self.macro_state_ensemble[3,:]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_hat = np.matrix([x_mean, y_mean]).T
        C = np.cov(x,y)
        # Set up grid for plotting
        x_grid = np.linspace(0, max(x), 100)
        y_grid = np.linspace(min(y)/2, max(y)*1.2, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        varname1 = "$ Wealth per adult Top 1%"
        varname2 = "$ Weatlh per adult Bottom 50%"
        plot_bivariate_normal(x_hat, C, X, Y, varname1, varname2)
        
    def plot_micro_state(self):
        pass
        
        ### plot distribution over all agents wealth as prob. density 
        ### from here we can also plot the kalman filter process with 
        ### its macro estimates on the 
        ### same plot
        
        #### micro state as one-dim prob distribution is only one system state.
        
        
        ### plot many overlapping probability distributions

    def step(self):
        self.predict()
        self.set_current_obs()
        self.update_state_ensemble()
        self.update_state_mean()
        self.update_data_ensemble()
        self.make_ensemble_covariance()
        self.make_data_covariance()
        self.make_gain_matrix()
        #self.state_update()

