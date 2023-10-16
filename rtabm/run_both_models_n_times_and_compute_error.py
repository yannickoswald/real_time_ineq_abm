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


class benchmarking_error:
    
    def __init__(self, ensemble_size, **kwargs):
        
        self.model1_data = list()
        self.model2_data = list()
        self.ensemble_size = ensemble_size
        ### LOAD and PREPARE DATA
        path = ".."
        with open(os.path.join(path, 'data', 'wealth_data_for_import2.csv')) as f2:
            self.df = pd.read_csv(f2, encoding = 'unicode_escape')   
        
        ### Set up variables for data storage and passing between methods
        self.subset_df = None
        self.mean_error_model1 = None
        self.mean_error_model2 = None

    # ERROR computing function
    def quantify_error(self, model_output, data_vector):
        
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


    def collect_data(self):
     
       """
       RUN MODEL n times and collect data
       
       """
       ## run models n times and write out data 
       for i in range(self.ensemble_size):
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
            self.model1_data.append(economy1.plot_wealth_groups_over_time(ax1, time_horizon))
            self.model2_data.append(economy2.plot_wealth_groups_over_time(ax2))

    def compute_error(self):
        
        df = self.df
        # Subset the dataframe
        subset_df = df[((df['year'] == 1990) & (df['month'] == 'Jan')) |
                       ((df['year'] > 1989) & (df['year'] < 2019)) |
                       ((df['year'] == 2018) & (df['month'] == 'Dec'))]
        self.subset_df = subset_df
        # Extract data for each group and store in a list of arrays
        arrays = [subset_df[subset_df['group'] == grp]['real_wealth_share'].to_numpy() for grp in subset_df['group'].unique()]
        # Get the maximum length among the arrays
        max_len = max(len(arr) for arr in arrays)
        # Ensure all arrays are of the same length by appending a specific value (like np.nan) to shorter arrays
        arrays = [np.concatenate([arr, [np.nan]*(max_len - len(arr))]) for arr in arrays]
        # Horizontally stack arrays to get the desired result
        data_array = np.column_stack(arrays)

        ## use numpy mean 
        errors_model1 = np.zeros((data_array.shape[0], self.ensemble_size))
        errors_model2 = np.zeros((data_array.shape[0], self.ensemble_size))
        
        for i in range(self.ensemble_size):
            errors_model1[:,i] = self.quantify_error(self.model1_data[i], data_array)
            errors_model2[:,i] = self.quantify_error(self.model2_data[i], data_array)
            
        self.mean_error_model1 = np.mean(errors_model1,axis = 1)
        self.mean_error_model2 = np.mean(errors_model2,axis = 1)
    
    
    def plot_graph(self):
    
        fig, ax = plt.subplots(figsize=(10,4))
        x = self.subset_df["date_short"][::4].reset_index(drop = True)
        ax.plot(x, self.mean_error_model1, label = "model 1")
        ax.plot(x, self.mean_error_model2, label = "model 2")
        ax.set_xticks(x.iloc[0::20].index)
        ax.set_xticklabels(x.iloc[0::20], rotation = 90)
        ax.legend(frameon = False)
        ax.set_ylabel("error metric")
        ax.margins(0)
        plt.savefig('fig3.png',  bbox_inches='tight', dpi=300)
        
#%%
if __name__ == "__main__":
    benchmark = benchmarking_error(20)
    benchmark.collect_data()
    benchmark.compute_error()
    benchmark.plot_graph()
