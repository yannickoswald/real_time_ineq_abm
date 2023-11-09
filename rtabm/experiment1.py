# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:20:14 2023

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
from run_enkf2 import *
#from run_both_models_n_times_and_compute_error import *


#%%


class Experiment1:
    def __init__(self, num_agents, ensemble_size, macro_state_dim):
        self.num_agents = num_agents
        self.ensemble_size = ensemble_size
        self.macro_state_dim = macro_state_dim
        self.enkf1 = None
        self.enkf2 = None

    def run_both_enkf(self, time_horizon):
        self.enkf1 = prepare_enkf(self.num_agents, self.ensemble_size, self.macro_state_dim)
        self.enkf2 = prepare_enkf2(self.num_agents, self.ensemble_size, self.macro_state_dim)
        
        run_enkf(self.enkf1, time_horizon)
        run_enkf(self.enkf2, time_horizon)

    def plot_results(self):
        fig = plt.figure(figsize=(10, 10))
        # Create a gridspec object
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        # Create individual subplots
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[0, 1])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])
        ax4 = plt.subplot(gs[2, :])  # This one spans both columns

        
        self.enkf1.models[0].plot_wealth_groups_over_time(ax0, 29*12)
        self.enkf2.models[0].plot_wealth_groups_over_time(ax1, 29*12)
        self.enkf1.plot_fanchart(ax2)
        self.enkf2.plot_fanchart(ax3)        
        self.enkf1.plot_error(ax4)
        self.enkf2.plot_error(ax4)

        ###EXTRAS
        #AX0
        ax0.text(0, 0.85, 'a', fontsize = 12)
        ax0.text(40, 0.85, 'Model 1', fontsize = 12)
        #AX1
        ax1.legend(loc=(1.05, -0.15), frameon = False) ### legend only here
        ax1.text(0, 0.85, 'b', fontsize = 12)
        ax1.text(40, 0.85, 'Model 2', fontsize = 12)
        #AX2
        ax2.text(0, 1.05, 'c', fontsize = 12)
        ax2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax2.text(40,1.05, 'Model 1', fontsize = 12)
        #AX3
        ax3.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax3.text(0,1.05, 'd', fontsize = 12)
        ax3.text(40,1.05, 'Model 2', fontsize = 12)

        path = '..'
        with open(os.path.join(path, 'data', 'mean_errors.csv')) as f:
            errors_df_no_enkf = pd.read_csv(f, encoding='unicode_escape')

        ax4.plot(errors_df_no_enkf['mean_error_model1'], linestyle='--', label='Model 1 no ENKF', color='tab:blue')
        ax4.plot(errors_df_no_enkf['mean_error_model2'], linestyle='--', label='Model 2 no ENKF', color='tab:orange')
        ax4.legend(bbox_to_anchor=(1.05, 1), frameon=False)
        
        # Get the limits
        x_min, x_max = ax4.get_xlim()
        y_min, y_max = ax4.get_ylim()
        ax4.text(0, y_max+0.02, 'e', fontsize = 12)


        plt.tight_layout()
        #plt.savefig('fig4.png', dpi=300)
        plt.show()


# Example usage
if __name__ == "__main__":
    experiment1 = Experiment1(num_agents=100, ensemble_size=10, macro_state_dim=4)
    experiment1.run_both_enkf(time_horizon = 29*12)
    experiment1.plot_results()
'''
#if __name__=="__main__":
enkf1 = prepare_enkf(num_agents=100, ensemble_size=10, macro_state_dim=4)
enkf2 = prepare_enkf2(num_agents=100, ensemble_size=10, macro_state_dim = 4)

run_enkf(enkf1)
run_enkf(enkf2)

# Now let's say you want to integrate this into another grid layout
fig = plt.figure(figsize=(10, 10))
# Create a gridspec object
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
# Create individual subplots
ax0 = plt.subplot(gs[0, 0])
ax1 = plt.subplot(gs[0, 1])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[1, 1])
ax4 = plt.subplot(gs[2, :])  # This one spans both columns

enkf1.models[0].plot_wealth_groups_over_time(ax0, 29*12)
enkf2.models[0].plot_wealth_groups_over_time(ax1, 29*12)
enkf1.plot_fanchart(ax2)
enkf2.plot_fanchart(ax3)
enkf1.plot_error(ax4)
enkf2.plot_error(ax4)


### IMPORT AND PLOT ERRORS FROM NONE-ENKF RUNS
path = '..'
with open(os.path.join(path, 'data', 'mean_errors.csv')) as f:
    errors_df_no_enkf = pd.read_csv(f, encoding = 'unicode_escape')  

###EXTRAS
#AX0
ax0.text(0, 0.85, 'a', fontsize = 12)
ax0.text(40, 0.85, 'Model 1', fontsize = 12)
#AX1
ax1.legend(loc=(1.05, -0.15), frameon = False) ### legend only here
ax1.text(0, 0.85, 'b', fontsize = 12)
ax1.text(40, 0.85, 'Model 2', fontsize = 12)
#AX2
ax2.text(0, 1.05, 'c', fontsize = 12)
ax2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax2.text(40,1.05, 'Model 1', fontsize = 12)
#AX3
ax3.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax3.text(0,1.05, 'd', fontsize = 12)
ax3.text(40,1.05, 'Model 2', fontsize = 12)

#AX4
# Plotting the two columns with dashed lines
# Replace 'column1' and 'column2' with the actual column names from errors_df_no_enkf
ax4.plot(errors_df_no_enkf['mean_error_model1'], linestyle='--', label='Model 1 no ENKF', color = 'tab:blue')
ax4.plot(errors_df_no_enkf['mean_error_model2'], linestyle='--', label='Model 2 no ENKF', color = 'tab:orange')
ax4.legend(bbox_to_anchor=(1.05, 1), frameon=False)

# Get the limits
x_min, x_max = ax4.get_xlim()
y_min, y_max = ax4.get_ylim()
ax4.text(0, y_max+0.02, 'e', fontsize = 12)


plt.tight_layout()
plt.savefig('fig4.png', dpi = 300)

plt.show()
'''