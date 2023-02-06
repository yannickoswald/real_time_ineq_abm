# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 09:46:44 2023

@author: earyo
"""

### import necessary libraries
import os
import pandas as pd
import numpy as np
import random
os.chdir(".")
from agent_class_wealth import WealthAgent

#%%

class Economy():
    
    """A model of the wealth distribution in the economy. 
       Wealth is defined as the physical and financial assets someone holds
       and is distinct from the income or the consumption of a person."""
       
    def __init__(self, economy_wealth_size, population_size, growth_rate):
        
        ### set economy (global) attributes
        self.num_agents = population_size
        self.economy_wealth = economy_wealth_size
        self.growth_rate_economy = growth_rate
        self.increments = 10
        ### set economy agents
        self.agents = self.make_agents()
        
        ### set other model parameters
        self.time = 0       
        
    def make_agents(self):
        agents = list()
        for i in range(self.num_agents):
            agents.append(WealthAgent(i, self.economy_wealth / self.num_agents, self))
        return agents

    def grow(self):     
        self.time = self.time + 1
        self.help_var = self.economy_wealth
        self.economy_wealth = self.economy_wealth * (1 + self.growth_rate_economy)
        self.new_wealth = self.economy_wealth - self.help_var
          
    def choose_agent(self):
        weights = []
        for x in self.agents: 
            weights.append(x.wealth_share)
        return random.choices(self.agents, weights, k = 1)
    
        
    def distribute_wealth(self):
        for increment in range(self.increments):
            agent_j = self.choose_agent()[0]
            agent_j.wealth = agent_j.wealth + (self.new_wealth / self.increments)
            
            
    def recalculate_wealth_shares(self):
        for x in self.agents: 
            x.determine_wealth_share()
     
    def __repr__(self):
        return f"{self.__class__.__name__}('population size: {self.num_agents}'),('economy size: {self.economy_wealth}')"
        
        
