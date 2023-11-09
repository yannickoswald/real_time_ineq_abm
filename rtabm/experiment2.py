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
