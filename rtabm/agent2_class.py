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
from exponential_pareto_avg_distr import weighted_avg_exp_pareto_distr
from exponential_pareto_avg_distr import map_percentiles_weights
from exponential_pareto_avg_distr import uniform_sample

class Agent2:
    def __init__(self, id, model, adaptive_sensitivity, distribution):
        self.id = id

        if distribution == "Pareto_lognormal":
            scale_coeff = 150000
            self.wealth = float(powerlognorm.rvs(1.92, 2.08, size=1))*scale_coeff

        elif distribution == "exponential_pareto":
            sample_uniform = float(np.random.uniform(0, 1, 1))
            lower_bound = 0.4
            upper_bound = 0.9
            agent_wealth_sample = weighted_avg_exp_pareto_distr(percentiles_given = sample_uniform, lower_bound = lower_bound, upper_bound = upper_bound, alpha = 1.3, Temperature = 5)
            self.wealth = agent_wealth_sample
     
    
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
        neighbors = [n for n in model.graph.neighbors(self.id)]
        # Trade with agents that are reachable via a single intermediary
        for neighbor in neighbors:
                self.trade(other     = model.graph.nodes[neighbor]["agent"], 
                           concavity = model.concavity)
        ## update wealth in line with economy wide economic growth        
        self.wealth = self.wealth * (1 + model.growth_rate)
        

    def trade(self, other, concavity):
        """Perform a trade between this agent and another agent"""
        # The fraction of wealth to be traded is limited to the wealth of the poorer agent
        a = self.willingness_to_risk
        b = other.willingness_to_risk
        fraction = random.uniform(0, min(a*self.wealth, b*other.wealth))
        # The probability of winning is proportional to the number of links of the agent
        self_win_probability = ((self.num_links / (self.num_links + other.num_links))**concavity) + np.random.normal(0,self.model.uncertainty_para)
        if random.random() < self_win_probability:
            # Self wins the trade
            self.wealth += fraction
            other.wealth -= fraction
            ### cap risk preferences at 10% of own wealth
            self.willingness_to_risk = min(0.1, self.willingness_to_risk*(1+self.s)) 
            other.willingness_to_risk = other.willingness_to_risk*(1-other.s)
        else:
            # Other wins the trade
            self.wealth -= fraction
            other.wealth += fraction
            self.willingness_to_risk = self.willingness_to_risk*(1-self.s)
            other.willingness_to_risk = min(0.1, other.willingness_to_risk*(1+other.s))
            
    def count_links(self, model):
        """Count the number of links this agent has"""
        return len(list(model.graph.neighbors(self.id)))
        


