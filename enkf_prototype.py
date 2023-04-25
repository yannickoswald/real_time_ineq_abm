# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:15:43 2023

@author: earyo
"""

import numpy as np

class EnsembleKalmanFilter:
    def __init__(self, model, ensemble_size, observation_error, state_error):
        self.model = model
        self.ensemble_size = ensemble_size
        self.observation_error = observation_error
        self.state_error = state_error
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.state_means = np.zeros((self.state_dim,))
        self.ensemble = np.zeros((self.ensemble_size, self.state_dim))
        self.state_covariance = np.eye(self.state_dim)
        self.observation_covariance = np.eye(self.obs_dim) * observation_error ** 2
        self.weights = np.ones((self.ensemble_size,)) / self.ensemble_size

    def update(self, observations):
        # Sample ensemble states from the prior distribution
        self.ensemble = np.random.multivariate_normal(self.state_means, self.state_covariance, self.ensemble_size)

        # Run the agent-based model forward for each ensemble member
        for i in range(self.ensemble_size):
            self.model.set_state(self.ensemble[i])
            self.ensemble[i] = self.model.step()

        # Compute the ensemble mean and covariance
        self.state_means = np.average(self.ensemble, axis=0, weights=self.weights)
        self.state_covariance = np.cov(self.ensemble.T, ddof=1, aweights=self.weights)

        # Compute the predicted observation means and covariance for each ensemble member
        predicted_observations = np.zeros((self.ensemble_size, self.obs_dim))
        for i in range(self.ensemble_size):
            predicted_observations[i] = self.model.observe(self.ensemble[i])
        predicted_observation_means = np.average(predicted_observations, axis=0, weights=self.weights)
        predicted_observation_covariance = np.cov(predicted_observations.T, ddof=1, aweights=self.weights)

        # Compute the Kalman gain
        innovation_covariance = predicted_observation_covariance + self.observation_covariance
        kalman_gain = np.dot(self.state_covariance, np.linalg.inv(innovation_covariance))

        # Update the ensemble states with the observations
        for i in range(self.ensemble_size):
            innovation = observations - predicted_observations[i]
            self.ensemble[i] = self.ensemble[i] + np.dot(kalman_gain, innovation)
        self.state_means = np.average(self.ensemble, axis=0, weights=self.weights)
        self.state_covariance = np.cov(self.ensemble.T, ddof=1, aweights=self.weights)

    def get_state(self):
        return self.state_means
    
    
    
    
    
    
def ensemble_kalman_filter(model, observations, ensemble_size=100, inflation_factor=1.0):
    """
    This function implements an Ensemble Kalman Filter (EnKF) used with an agent-based model.
    
    Parameters:
    model (function): A function that takes a state vector as input and returns a predicted state vector
    observations (list): A list of observed state vectors
    ensemble_size (int): The number of ensemble members to use (default is 100)
    inflation_factor (float): The inflation factor to use for the covariance matrix (default is 1.0)
    
    Returns:
    list: A list of filtered state vectors
    """
    try:
        # Check if the model function is callable
        if not callable(model):
            raise TypeError("The model argument must be a callable function")
        
        # Check if the observations list is not empty
        if not observations:
            raise ValueError("The observations list cannot be empty")
        
        # Check if the ensemble size is greater than zero
        if ensemble_size <= 0:
            raise ValueError("The ensemble size must be greater than zero")
        
        # Initialize the ensemble
        ensemble = [model() for i in range(ensemble_size)]
        
        # Calculate the mean and covariance of the ensemble
        mean = sum(ensemble) / ensemble_size
        cov = np.cov(ensemble, rowvar=False)
        
        # Loop over the observations and update the ensemble
        for obs in observations:
            # Generate a new ensemble by perturbing the mean
            perturbed_ensemble = np.random.multivariate_normal(mean, cov, ensemble_size)
            
            # Evaluate the model for each member of the perturbed ensemble
            perturbed_predictions = [model(state) for state in perturbed_ensemble]
            
            # Calculate the mean and covariance of the perturbed predictions
            perturbed_mean = sum(perturbed_predictions) / ensemble_size
            perturbed_cov = np.cov(perturbed_predictions, rowvar=False)
            
            # Calculate the Kalman gain
            kalman_gain = np.dot(cov, np.linalg.inv(cov + perturbed_cov * inflation_factor))
            
            # Update the mean and covariance of the ensemble
            mean = mean + np.dot(kalman_gain, obs - perturbed_mean)
            cov = np.dot(np.identity(ensemble_size) - np.dot(kalman_gain, perturbed_cov), cov)
        
        # Return the filtered ensemble
        return [model(state) for state in perturbed_ensemble]
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return []
    
    
    