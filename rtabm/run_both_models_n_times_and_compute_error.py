# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:38:24 2023

@author: earyo
"""

#General packages
import os
import numpy as np
from tqdm import tqdm  ### package for progress bars
import matplotlib.pyplot as plt
### MODEL 1 infrastructure
from model1_class import Model1
from enkf_yo import EnsembleKalmanFilter
### MODEL 2 infrastructure
from model2_class import Model2
from enkf_yo2 import EnsembleKalmanFilter
import pandas as pd


#%% ERROR computing function

def quantify_error(model_output, data_vector):
    
    """
    Compute the error metric as the average absolute distance between 
    the model output and the data vector

    :param model_output: 2D numpy array of shape [n, 4]
    :param data_vector: 2D numpy array of shape [n, 4]
    :return: float, the average error metric
    """
    # Ensure dimensions are correct
    assert model_output.shape == (model_output.shape[0], 4), "Model output should have shape [n, 4]"
    assert data_vector.shape == (data_vector.shape[0], 4), "Data vector should have shape [n, 4]"
    # Calculate absolute differences between the model output and data vector
    abs_diffs = np.abs(model_output - data_vector)    
    # sum differences across four wealth groups as in equation 6 of the paper
    abs_diffs_sum = np.sum(abs_diffs, axis = 1)
    # Return the average absolute difference as well as the error per group
    return abs_diffs_sum


#%%   PLOT FIGURE 3 showing error metric of n typical model run of both models 

##### RUN MODEL n times and collect data
model1_data = list()
model2_data = list()

n = 2 ## run models n times and write out data 

for i in range(n):
    # Set up both model economies
    economy1 = Model1(population_size=500,
                      growth_rate=0.025, 
                      b_begin=1.3,
                      distribution="Pareto_lognormal",
                      start_year=1990)

    economy2 = Model2(500,
                  concavity=1,
                  growth_rate = 0.025, 
                  start_year = 1990,
                  adaptive_sensitivity=0.02)

    ## define time horizon
    time_horizon = 29*12 ## 29 years * 12 months | from Jan 1990 to Dec 2018
    ### initialize model 1
    economy1.make_agents()
    ### run the models
    for i in tqdm(range(time_horizon)):
        economy1.step()
        economy2.step()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4))
    model1_data.append(economy1.plot_wealth_groups_over_time(ax1, time_horizon))
    model2_data.append(economy2.plot_wealth_groups_over_time(ax2))


#%% COMPUTE DATA ERROR ACROSS BOTH MODELS AND PLOT

### LOAD and PREPARE DATA
path = ".."
with open(os.path.join(path, 'data', 'wealth_data_for_import2.csv')) as f2:
    df = pd.read_csv(f2, encoding = 'unicode_escape')    
# Subset the dataframe
subset_df = df[((df['year'] == 1990) & (df['month'] == 'Jan')) |
               ((df['year'] > 1989) & (df['year'] < 2019)) |
               ((df['year'] == 2018) & (df['month'] == 'Dec'))]
# Extract data for each group and store in a list of arrays
arrays = [subset_df[subset_df['group'] == grp]['real_wealth_share'].to_numpy() for grp in subset_df['group'].unique()]
# Get the maximum length among the arrays
max_len = max(len(arr) for arr in arrays)
# Ensure all arrays are of the same length by appending a specific value (like np.nan) to shorter arrays
arrays = [np.concatenate([arr, [np.nan]*(max_len - len(arr))]) for arr in arrays]
# Horizontally stack arrays to get the desired result
data_array = np.column_stack(arrays)


## use numpy mean 
errors_model1 = np.zeros((data_array.shape[0], n))
errors_model2 = np.zeros((data_array.shape[0], n))


for i in range(n):
    errors_model1[:,i] = quantify_error(model1_data[i], data_array)
    errors_model2[:,i] = quantify_error(model2_data[i], data_array)
    
mean_error_model1 = np.mean(errors_model1,axis = 1)
mean_error_model2 = np.mean(errors_model2,axis = 1)


#%%   PLOT FIGURE 2 showing typical model run of both models 
fig, ax = plt.subplots(figsize=(10,4))
x = subset_df["date_short"][::4].reset_index(drop = True)
ax.plot(x, mean_error_model1, label = "model 1")
ax.plot(x, mean_error_model2, label = "model 2")

ax.set_xticks(x.iloc[0::20].index)
ax.set_xticklabels(x.iloc[0::20], rotation = 90)
ax.legend(frameon = False)
ax.set_ylabel("error metric")
plt.savefig('fig3.png',  bbox_inches='tight', dpi=300)
