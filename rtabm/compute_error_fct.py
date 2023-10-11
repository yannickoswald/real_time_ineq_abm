# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:14:22 2023

@author: earyo
"""

''' this script sets up a function that computes error between model 
(ENKF supported and without) and observation '''

import numpy as np

def compute_error_metric(model_output, data_vector):
    """
    Compute the error metric as the average absolute distance between 
    the model output and the data vector, averaged across all ensemble members 
    and the four wealth groups.

    :param model_output: 2D list or numpy array of shape [n, 4]
        where n is the number of ensemble members.
    :param data_vector: 1D list or numpy array of shape [4]
    :return: float, the error metric
    """
    
    # Convert to numpy arrays for easier calculations
    model_output = np.array(model_output)
    data_vector = np.array(data_vector)
    
    # Ensure dimensions are correct
    assert model_output.shape[1] == 4, "Model output should have shape [n, 4]"
    assert len(data_vector) == 4, "Data vector should have shape [4]"
    
    # Calculate absolute differences between the model output and data vector
    abs_diffs = np.abs(model_output - data_vector)
    
    # Return the average absolute difference
    return abs_diffs.mean()

# Example usage:
model_out = [[0.2, 0.35, 0.5, 0.65], 
             [0.22, 0.34, 0.51, 0.66], 
             [0.21, 0.36, 0.49, 0.64]]  # 3 ensemble members with predictions for 4 wealth groups
data_vec = [0.21, 0.35, 0.50, 0.65]

print(compute_error_metric(model_out, data_vec))