# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:03:32 2023

@author: earyo
"""

#General packages
import os
import numpy as np
from tqdm import tqdm  ### package for progress bars
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
### MODEL 1 infrastructure
import pandas as pd
from model1_class import Model1
from run_enkf import *
### MODEL 2 infrastructure
from model2_class import Model2
#from run_both_models_n_times_and_compute_error import *


#%%
''' this experiment investigate the influence 
of the ensemble size on the ENKF performance '''

##### for this experiment change the test period to a much shorter one
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming 'prepare_enkf', 'prepare_enkf2', 'run_enkf', and 'run_enkf2' are defined elsewhere

class Enkf_experiment5:
    
    def __init__(self, num_agents, macro_state_dim, repetitions, ensemble_size, uncertainty_in_models, uncertainty_obs_values, filter_frequency):
        self.num_agents = num_agents
        self.macro_state_dim = macro_state_dim
        self.repetitions = repetitions
        self.ensemble_size = ensemble_size
        self.uncertainty_obs_values = uncertainty_obs_values
        self.uncertainty_in_models = uncertainty_models
        self.filter_frequency = filter_frequency
        self.results_model1 = np.zeros((len(self.uncertainty_in_models), len(self.uncertainty_obs_values)))
        self.results_model2 = np.zeros((len(self.uncertainty_in_models), len(self.uncertainty_obs_values)))
        
    def run_experiment(self):
        
        #### set up result saving arrays
        #array_of_results_enkf1 = np.zeros((len(self.uncertainty_in_models), len(self.uncertainty_obs_values)))
        #array_of_results_enkf2 = np.zeros((len(self.uncertainty_in_models), len(self.uncertainty_obs_values)))
        
        
        #for idx, uncertainty_models in enumerate(tqdm(self.uncertainty_in_models)):
        for idx, uncertainty_models in enumerate(self.uncertainty_in_models): 
            #print(idx, uncertainty_models)
            #print("this is idx", uncertainty_models)
            
            model_params1 = {"population_size": 100,
             "growth_rate": 0.025,
             "b_begin": 1.3,
             "distribution": "Pareto_lognormal",
             "start_year": 1990,
             "uncertainty_para": uncertainty_models[0]}
            
            model_params2 = {"population_size": 100, 
                            "concavity": 1,
                            "growth_rate": 0.025, 
                            "start_year": 1990,
                            "adaptive_sensitivity": 0.02,
                            "uncertainty_para": uncertainty_models[1]}
            
            for jdx, uncertainty_obs in enumerate(self.uncertainty_obs_values):
                
                    #print("this is jdx", uncertainty_obs)
                    #print(jdx, uncertainty_obs)
                
                    array_of_results_enkf1_repetitions = np.zeros((self.repetitions, 1))
                    array_of_results_enkf2_repetitions = np.zeros((self.repetitions, 1))
                    #print("this is shape", array_of_results_enkf1_repetitions.shape)
                    for i in range(self.repetitions):
                        
                        enkf1 = prepare_enkf(Model= Model1, model_params = model_params1, 
                                             ensemble_size=self.ensemble_size,
                                             macro_state_dim=self.macro_state_dim,
                                             uncertainty_obs=uncertainty_obs,
                                             filter_freq= self.filter_frequency)
                        
                        enkf2 = prepare_enkf(Model= Model2, model_params = model_params2,
                                             ensemble_size=self.ensemble_size, 
                                             macro_state_dim=self.macro_state_dim,
                                             uncertainty_obs=uncertainty_obs,
                                             filter_freq= self.filter_frequency)
                        
                        run_enkf(enkf1, time_horizon=12*3, filter_freq=10)
                        run_enkf(enkf2, time_horizon=12*3, filter_freq=10)
                        enkf1.make_macro_history_share()
                        enkf2.make_macro_history_share()
                        array_of_results_enkf1_repetitions[i, 0] = enkf1.post_update_difference()
                        array_of_results_enkf2_repetitions[i, 0] = enkf2.post_update_difference()
            
                        
                    #print("this is (np.mean(array_of_results_enkf1_repetitions))", np.mean(array_of_results_enkf1_repetitions))
                    #print("this is np.mean(array_of_results_enkf2_repetitions", np.mean(array_of_results_enkf2_repetitions))
                    print("this is idx, jdx", idx, jdx)
                    self.results_model1[idx, jdx] = np.mean(array_of_results_enkf1_repetitions) ## how this is setup idx goes down the rows, which means along the vertical dimension idx = model varies
                    self.results_model2[idx, jdx] = np.mean(array_of_results_enkf2_repetitions)
                            
             
    def plot_heatmap(self, results, model_name, ax=None, save_fig=False, fig_name='fig9.png'):
        # Define the levels of contours
        levels = np.linspace(np.min(results), np.max(results), num=20)
    
        # If an Axes object isn't provided, create a new figure and Axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
    
        # Create the contour plot on the provided Axes
        contour = ax.contourf(results, levels=levels, cmap='plasma', extend='neither')
        cbar = plt.colorbar(contour, ax=ax, label='Average Error', extend='neither')
        
        # Set the labels for ticks if needed
        
        labels_yaxis = [x[0] for x in self.uncertainty_in_models]
    
        ax.set_xticks(ticks=np.arange( len(self.uncertainty_obs_values)), labels=self.uncertainty_obs_values)
        ax.set_yticks(ticks=np.arange(len(self.uncertainty_in_models)), labels=labels_yaxis)
    
        # Set the labels and title
        ax.set_xlabel('Uncertainty data', fontsize = 16) # see comment for self.results_model1
        ax.set_ylabel('Uncertainty model', fontsize = 16)
        ax.set_title(f'ENKF Performance Heatmap for {model_name}', fontsize = 16)
        
        # Save the figure if requested and if no specific Axes is provided
        if save_fig and ax is None:
            plt.savefig(f"{model_name}_{fig_name}", dpi=300)
    
        # Return the Axes object
        return ax

        
# Usage of the class
num_agents = 100
macro_state_dim = 4
repetitions = 5
ensemble_size = 20  # Example ensemble size
filter_frequency = 10
uncertainty_obs = [0.05, 0.1, 0.4, 0.8, 1] ###[0.01, 0.1, 0.5, 1] #[0.1, 0.5, 2, 10]  # Example values for uncertainty_obs
uncertainty_models = [(0.1, 0.1), (0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (1, 1)]#[(0.1, 0.1), (0.2, 0.2), (0.5, 0.5), (0.8, 0.8)]
experiment = Enkf_experiment5(num_agents,
                              macro_state_dim,
                              repetitions, 
                              ensemble_size,
                              uncertainty_in_models = uncertainty_models,
                              uncertainty_obs_values = uncertainty_obs,
                              filter_frequency=filter_frequency)
experiment.run_experiment()

# Create a figure for the subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot heatmap for Model 1 in the first subplot (ax1)
experiment.plot_heatmap(experiment.results_model1, "Model1", ax=ax1, save_fig=False)
ax1.set_title('Model1', fontsize = 16)
ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Plot heatmap for Model 2 in the second subplot (ax2)
experiment.plot_heatmap(experiment.results_model2, "Model2", ax=ax2, save_fig=False)
ax2.set_title('Model2', fontsize = 16)
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the entire figure
plt.savefig('fig8.png', dpi=300)

# Show the plot
plt.show()


