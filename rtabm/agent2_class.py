# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:41:18 2023

@author: earyo
"""


import networkx as nx
import matplotlib.pyplot as plt
import random as random
from inequality_metrics import find_wealth_groups2
import numpy as np
from scipy.stats import powerlognorm
import pandas as pd
import os
from exponential_pareto_avg_distr import weighted_avg_exp_pareto_distr
from exponential_pareto_avg_distr import map_percentiles_weights
from exponential_pareto_avg_distr import uniform_sample

class Agent2:
    def __init__(self, id, model, adaptive_sensitivity, distribution):
        self.id = id
        self.transaction_history = []

        if distribution == "Pareto_lognormal":
            scale_coeff = 150000
            self.wealth = float(powerlognorm.rvs(1.92, 2.08, size=1))*scale_coeff

        elif distribution == "exponential_pareto":

             # load data to find the average wealth of and scalefactor 
            path = ".."
            with open(os.path.join(path, 'data', 'average_wealth_for_every_year.csv')) as f:
                 d_average = pd.read_csv(f,  encoding='utf-8-sig', sep = ",")

            # Rename the column if it has unexpected characters
            d_average.rename(columns={'ï»¿Year': 'Year'}, inplace=True)

            # subset only the data where the Month is equal 1 (i.e. the first month of the year)
            d_average = d_average[d_average["Month"] == 1]

            # print head of the data
            #print(d_average.head())

            # find the average wealth for the year the model starts
            average_wealth = d_average[d_average["Year"] == model.start_year]["Real Wealth Per Unit"].values[0]
            # find scale factor for the year the model starts per formula derived in test calibration new weighted avg
            scale_factor = 0.04*average_wealth ## applied equation from test_calibration_new_weighted_avg
         

            sample_uniform = float(np.random.uniform(0, 1, 1))
            lower_bound = 0.4
            upper_bound = 0.9
            agent_wealth_sample = weighted_avg_exp_pareto_distr(percentiles_given = sample_uniform, lower_bound = lower_bound, upper_bound = upper_bound, alpha = 1.3, Temperature = 5)
            self.wealth = agent_wealth_sample * scale_factor
            #print(f"agent wealth: {np.log(self.wealth)}")
     
    
        self.model = model
        ### variable that determines how much an agent is willing to trade/risk
        self.willingness_to_risk = random.uniform(0, 0.1)
        self.num_links = 1 ### placeholder only. is calculated in assigned method
        ### parameter that determines by how many % after a trade an agent
        ### adjusts their risk preferences
        self.s = adaptive_sensitivity

    def step(self, model):
        """Trade with linked agents"""
        # Count own relationships/links
        self.num_links = self.count_links(model = self.model)
        # Get the neighbors of this agent
        neighbors = [n for n in model.graph.neighbors(self.id) if n != self.id]
        # Trade with agents that are reachable via a single intermediary
        for neighbor in neighbors:
                self.trade(other     = model.graph.nodes[neighbor]["agent"], 
                           concavity = model.concavity) ### concavity is a parameter that determines how much the number of links influences the probability of winning a trade
        ## update wealth in line with economy wide economic growth
        self.wealth = self.wealth * (1 + model.growth_rate) # remember this is monthly growth rate
        

    def trade(self, other, concavity):
        """Perform a trade between this agent and another agent"""

        # trade only if both agents have positive wealth
        # if self.wealth <= 0 or other.wealth <= 0:
           #  return

        # The fraction of wealth to be traded is limited to the wealth of the poorer agent
        a = self.willingness_to_risk
        b = other.willingness_to_risk
        fraction = random.uniform(0, min(a*self.wealth, b*other.wealth))
        # The probability of winning is proportional to the number of links of the agent compared to the other agent
        self_win_probability = ((self.num_links / (self.num_links + other.num_links))**concavity) + np.random.normal(0,self.model.uncertainty_para)
        # ensure self_win_probability is between 0 and 1
        self_win_probability = min(1, max(0, self_win_probability))
        #print("this is a trade fraction, agent ids are, ", self.id,',', other.id, 'time is, ', self.model.time,',', 'fraction,', fraction)
        self.transaction_history.append((self.id, other.id, self.model.time, fraction[0] if isinstance(fraction, (list, np.ndarray)) else fraction))

         # Record the transaction in both agents' histories
        transaction = (self.id, other.id, self.model.time, fraction[0] if isinstance(fraction, (list, np.ndarray)) else fraction)
        self.transaction_history.append(transaction)
        other.transaction_history.append(transaction)  

        if random.random() < self_win_probability:
            # Self wins the trade
            self.wealth += fraction
            other.wealth -= fraction
            # if other.wealth < 0:
              #   other.wealth = 0
            ### cap risk preferences at 10% of own wealth
            self.willingness_to_risk = min(0.1, self.willingness_to_risk*(1+self.s)) 
            other.willingness_to_risk = other.willingness_to_risk*(1-other.s)

        else:
            # Other wins the trade
            self.wealth -= fraction
            # if self.wealth < 0:
               #  self.wealth = 0
            other.wealth += fraction
            self.willingness_to_risk = self.willingness_to_risk*(1-self.s)
            other.willingness_to_risk = min(0.1, other.willingness_to_risk*(1+other.s))
            
    def count_links(self, model):
        """Count the number of links this agent has"""
        return len(list(model.graph.neighbors(self.id)))
        


