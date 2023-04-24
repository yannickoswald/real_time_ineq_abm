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