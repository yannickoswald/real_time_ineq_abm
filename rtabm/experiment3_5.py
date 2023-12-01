# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:20:41 2023

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


class Enkf_experiment4:

    def __init__(self, num_agents, macro_state_dim, repetitions, ensemble_sizes, filter_frequencies):
        self.num_agents = num_agents
        self.macro_state_dim = macro_state_dim
        self.repetitions = repetitions
        self.ensemble_sizes = ensemble_sizes
        self.filter_frequencies = filter_frequencies
        self.results = np.zeros((len(ensemble_sizes), len(filter_frequencies)))

    def run_experiment(self):
        for i, size in enumerate(self.ensemble_sizes):
            for j, freq in enumerate(self.filter_frequencies):
                error_sum_enkf1 = 0
                error_sum_enkf2 = 0
                for _ in tqdm(range(self.repetitions), desc=f"Size {size}, Freq {freq}"):
                    enkf1 = prepare_enkf(num_agents=self.num_agents, ensemble_size=size, macro_state_dim=self.macro_state_dim)
                    enkf2 = prepare_enkf2(num_agents=self.num_agents, ensemble_size=size, macro_state_dim=self.macro_state_dim)
                    run_enkf(enkf1, time_horizon=12*3, filter_freq=freq)
                    run_enkf2(enkf2, time_horizon=12*3, filter_freq=freq)
                    error_sum_enkf1 += enkf1.integral_error()
                    error_sum_enkf2 += enkf2.integral_error()
                self.results[i, j] = (error_sum_enkf1 + error_sum_enkf2) / (2 * self.repetitions)

    def plot_heatmap(self, save_fig=False, fig_name='heatmap.png'):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.results, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Average Error')
        plt.xticks(ticks=np.arange(len(self.filter_frequencies)), labels=self.filter_frequencies)
        plt.yticks(ticks=np.arange(len(self.ensemble_sizes)), labels=self.ensemble_sizes)
        plt.xlabel('Ensemble Size')
        plt.ylabel('Filter Frequency')
        plt.title('ENKF Performance Heatmap')
        if save_fig:
            plt.savefig(fig_name, dpi=300)
        plt.show()

# Usage
experiment35 = Enkf_experiment4(num_agents=100, macro_state_dim=4, repetitions=20,
                                 ensemble_sizes=[5, 10, 15, 20, 25, 30], filter_frequencies=[2, 5, 10, 20, 50, 100])
experiment35.run_experiment()
experiment35.plot_heatmap(save_fig=True)

