# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:20:33 2023

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
''' this experiment investigate the influence 
of the ensemble size on the ENKF performance '''

##### for this experiment change the test period to a much shorter one
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming 'prepare_enkf', 'prepare_enkf2', 'run_enkf', and 'run_enkf2' are defined elsewhere

class Enkf_experiment3:
    
    def __init__(self, num_agents, macro_state_dim, repetitions, ensemble_sizes):
        self.num_agents = num_agents
        self.macro_state_dim = macro_state_dim
        self.repetitions = repetitions
        self.ensemble_sizes = ensemble_sizes
        self.results = []
        
    def run_experiment(self):
        for size in self.ensemble_sizes:
            array_of_results_enkf1 = np.zeros((self.repetitions, 1))
            array_of_results_enkf2 = np.zeros((self.repetitions, 1))
            for i in tqdm(range(self.repetitions), desc=f"Ensemble size {size} repetitions"):
                enkf1 = prepare_enkf(num_agents=self.num_agents, ensemble_size=size, macro_state_dim=self.macro_state_dim)
                enkf2 = prepare_enkf2(num_agents=self.num_agents, ensemble_size=size, macro_state_dim=self.macro_state_dim)
                run_enkf(enkf1)
                run_enkf2(enkf2)
                array_of_results_enkf1[i, 0] = enkf1.integral_error()
                array_of_results_enkf2[i, 0] = enkf2.integral_error()
                
            self.results.append([f'Ensemble size {size}', array_of_results_enkf1, array_of_results_enkf2])
    
    def plot_results(self, save_fig=False, fig_name='fig5.png'):
        fig, ax = plt.subplots()
        boxplot_artists = []
        positions = np.arange(len(self.results)) * 2

        for i, (label, array1, array2) in enumerate(self.results):
            pos = positions[i] + np.array([-0.4, 0.4])
            box1 = ax.boxplot(array1, positions=[pos[0]], widths=0.6, patch_artist=True)
            box2 = ax.boxplot(array2, positions=[pos[1]], widths=0.6, patch_artist=True)
            plt.setp(box1["boxes"], facecolor='lightcoral')
            plt.setp(box1["medians"], color='black')
            plt.setp(box2["boxes"], facecolor='lightblue')
            plt.setp(box2["medians"], color='black')
            ax.plot(np.random.normal(pos[0], 0.04, size=len(array1)), array1, 'r.', alpha=0.7)
            ax.plot(np.random.normal(pos[1], 0.04, size=len(array2)), array2, 'b.', alpha=0.7)
            if i == 0:
                boxplot_artists.append(box1['boxes'][0])
                boxplot_artists.append(box2['boxes'][0])
                
        ax.set_xticks(positions)
        ax.set_xticklabels([label for label, _, _ in self.results])
        ax.set_ylabel('Error sum under curve of mean error')
        legend_labels = ['Model1 ENKF Boxplot', 'Model 2 ENKF Boxplot']
        ax.legend(boxplot_artists, legend_labels, title='Legend', loc="upper right", frameon=False)

        plt.tight_layout()
        if save_fig:
            plt.savefig(fig_name, dpi=300)
        plt.show()
        
        
    def compute_elasticity_fit(self):
    
        means1 = np.zeros((len(self.results), 1))
        means2 = np.zeros((len(self.results), 1))

        for idx, (label, array1, array2) in enumerate(self.results):
            means1[idx, 0] = np.mean(array1)
            means2[idx, 0] = np.mean(array2)

       # print(means1)
        #print(means2)

        # Transform to log space for linear regression
        log_means1 = np.log(means1)
        log_means2 = np.log(means2)
        log_ensemble_sizes = np.log(self.ensemble_sizes)

        # Perform the linear regression in log space
        # The slope will be the exponent b and the intercept will be log(a)
        slope1, intercept1 = np.polyfit(log_ensemble_sizes, log_means1.ravel(), 1)
        slope2, intercept2 = np.polyfit(log_ensemble_sizes, log_means2.ravel(), 1)

        # Convert intercept into the coefficient a in the original space
        a1 = np.exp(intercept1)
        a2 = np.exp(intercept2)
        
        # Return the coefficients (a, b) for both fits
        return (a1, slope1), (a2, slope2)

# To use the class
num_agents = 100
macro_state_dim = 4
repetitions = 20
ensemble_sizes = [5,10,30,100]

experiment = Enkf_experiment3(num_agents, macro_state_dim, repetitions, ensemble_sizes)
experiment.run_experiment()
experiment.plot_results(save_fig=True)
elasticities = experiment.compute_elasticity_fit()



'''
repetitions = 10
list_of_results_over_repetitions = list()

### values we test 
list_of_ensemble_size = [5, 10, 30, 100]

for x in list_of_ensemble_size: #tqdm(list_of_ensemble_size, desc = f"Current ensemble size is {ensemble_size}"):
    array_of_results_enkf1 = np.zeros((repetitions, 1))
    array_of_results_enkf2 = np.zeros((repetitions, 1))
    for i in tqdm(range(repetitions), desc = "Current ensemble size repetitions"):
        
        enkf1 = prepare_enkf(num_agents=100, ensemble_size=x, macro_state_dim=4)
        enkf2 = prepare_enkf2(num_agents=100, ensemble_size=x, macro_state_dim = 4)
        run_enkf(enkf1)
        run_enkf2(enkf2)
        array_of_results_enkf1[i,0] = enkf1.integral_error()
        array_of_results_enkf2[i,0] = enkf2.integral_error()
        
    list_of_results_over_repetitions.append([f' Ensemble size {x}',
                                             array_of_results_enkf1,
                                             array_of_results_enkf2])
        
        
        
# Let's create a mock data set as per the user's description
# Each sublist has a description and two arrays of data
data = list_of_results_over_repetitions
# Initialize the plot
fig, ax = plt.subplots()

# We will store the artists for the legend
boxplot_artists = []

# Position on the x-axis for the boxplots
positions = np.arange(len(data)) * 2


for i, (label, array1, array2) in enumerate(data):
    # Calculate positions for the two boxplots
    pos = positions[i] + np.array([-0.4, 0.4])
    
    # Create the boxplots
    box1 = ax.boxplot(array1, positions=[pos[0]], widths=0.6, patch_artist=True)
    box2 = ax.boxplot(array2, positions=[pos[1]], widths=0.6, patch_artist=True)
    
    # Set properties for the boxplots for clarity
    plt.setp(box1["boxes"], facecolor='lightcoral')
    plt.setp(box1["medians"], color='black')
    plt.setp(box2["boxes"], facecolor='lightblue')
    plt.setp(box2["medians"], color='black')
    
    # Plot the data points as dots
    ax.plot(np.random.normal(pos[0], 0.04, size=len(array1)), array1, 'r.', alpha=0.7)
    ax.plot(np.random.normal(pos[1], 0.04, size=len(array2)), array2, 'b.', alpha=0.7)

    # We only need to add one sample of each boxplot type to the legend
    if i == 0:
        boxplot_artists.append(box1['boxes'][0])
        boxplot_artists.append(box2['boxes'][0])
    
# Set the x-axis labels to the group descriptions
ax.set_xticks(positions)
ax.set_xticklabels([label for label, _, _ in data])

# Add some labels and a title
ax.set_ylabel('Error sum under curve of mean error')

# Create legend for boxplots only
legend_labels = ['EnKF1 Boxplot', 'EnKF2 Boxplot']
ax.legend(boxplot_artists, legend_labels, title='Legend', loc = "upper right", frameon=False)


plt.tight_layout()
plt.savefig('fig5.png', dpi = 300)


# Show the plot
plt.show()
    '''
'''
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



import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming 'prepare_enkf', 'prepare_enkf2', 'run_enkf', and 'run_enkf2' are defined elsewhere

class EnKFExperiment:
    def __init__(self, num_agents, macro_state_dim, repetitions, ensemble_sizes):
        self.num_agents = num_agents
        self.macro_state_dim = macro_state_dim
        self.repetitions = repetitions
        self.ensemble_sizes = ensemble_sizes
        self.results = []
        
    def run_experiment(self):
        for size in self.ensemble_sizes:
            array_of_results_enkf1 = np.zeros((self.repetitions, 1))
            array_of_results_enkf2 = np.zeros((self.repetitions, 1))
            for i in tqdm(range(self.repetitions), desc=f"Ensemble size {size} repetitions"):
                enkf1 = prepare_enkf(num_agents=self.num_agents, ensemble_size=size, macro_state_dim=self.macro_state_dim)
                enkf2 = prepare_enkf2(num_agents=self.num_agents, ensemble_size=size, macro_state_dim=self.macro_state_dim)
                run_enkf(enkf1)
                run_enkf2(enkf2)
                array_of_results_enkf1[i, 0] = enkf1.integral_error()
                array_of_results_enkf2[i, 0] = enkf2.integral_error()
                
            self.results.append([f'Ensemble size {size}', array_of_results_enkf1, array_of_results_enkf2])
    
    def plot_results(self, save_fig=False, fig_name='fig5.png'):
        fig, ax = plt.subplots()
        boxplot_artists = []
        positions = np.arange(len(self.results)) * 2

        for i, (label, array1, array2) in enumerate(self.results):
            pos = positions[i] + np.array([-0.4, 0.4])
            box1 = ax.boxplot(array1, positions=[pos[0]], widths=0.6, patch_artist=True)
            box2 = ax.boxplot(array2, positions=[pos[1]], widths=0.6, patch_artist=True)
            plt.setp(box1["boxes"], facecolor='lightcoral')
            plt.setp(box1["medians"], color='black')
            plt.setp(box2["boxes"], facecolor='lightblue')
            plt.setp(box2["medians"], color='black')
            ax.plot(np.random.normal(pos[0], 0.04, size=len(array1)), array1, 'r.', alpha=0.7)
            ax.plot(np.random.normal(pos[1], 0.04, size=len(array2)), array2, 'b.', alpha=0.7)
            if i == 0:
                boxplot_artists.append(box1['boxes'][0])
                boxplot_artists.append(box2['boxes'][0])
                
        ax.set_xticks(positions)
        ax.set_xticklabels([label for label, _, _ in self.results])
        ax.set_ylabel('Error sum under curve of mean error')
        legend_labels = ['EnKF1 Boxplot', 'EnKF2 Boxplot']
        ax.legend(boxplot_artists, legend_labels, title='Legend', loc="upper right", frameon=False)

        plt.tight_layout()
        if save_fig:
            plt.savefig(fig_name, dpi=300)
        plt.show()

# To use the class
num_agents = 100
macro_state_dim = 4
repetitions = 10
ensemble_sizes = [5, 10, 30, 100]

experiment = EnKFExperiment(num_agents, macro_state_dim, repetitions, ensemble_sizes)
experiment.run_experiment()
experiment.plot_results(save_fig=True)



'''