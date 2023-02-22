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
from scipy.stats import powerlognorm

#%%

class Economy():
    
    """A model of the wealth distribution in the economy. 
       Wealth is defined as the physical and financial assets someone
       (or society as a whole) holds and is distinct from the income 
       or the consumption of a person."""
       
    def __init__(self, 
                 economy_wealth_size,
                 population_size,
                 growth_rate,
                 b_begin,
                 distribution: str):
        
        ### set economy (global) attributes
        self.num_agents = population_size
        
        self.growth_rate_economy = (1+growth_rate)**(1/365) - 1## ~growth_rate / 365 ### DAILY growth rate
        #### the number of increments is important since it determines how the new
        #### wealth growth is divided and how many chances there are to receive some. 
        self.increments = 10
        
        ### SET OTHER MODEL PARAMETERS
        ### track time in the economy 
        self.time = 0    
        
        ### The sum_power is a model parameter which helps calculate the 
        ### normalized wealth share of
        ### an agent given the agent parameter beta. Initialized as the usual 
        ### sum of wealth
        self.sum_power = economy_wealth_size
        ### Beta is an economy wide 
        ### scaling parameter of how power to receive more wealth scales with 
        ### wealth itself.
        self.economy_beta = b_begin
        
        
        ### set distribution from which agents' wealth is sampled initially
        if distribution == "all_equal":
            self.economy_wealth = economy_wealth_size
            self.distr = self.economy_wealth / self.num_agents
        elif distribution == "Pareto_lognormal":
            ### the scaling_coefficient is determined through the actual wealth average
            ### in USD which is ~ 4.1*10^5 and the average/mean of the standardized
            ### pareto lognormal distr which then still has to be scaled to match
            ### the empirical distribution
            scaling_coefficient = 410000 /5.26
            self.distr =  powerlognorm.rvs(1.05, 1.9, size=1)*scaling_coefficient
        
        ### set economy agents
        self.agents = self.make_agents()
        
     
        
        
    def make_agents(self):
        agents = list()
        for i in range(self.num_agents):
            ### The agent parameters are: i = unique_id, economy_wealth / num_agents =
            ### every agent receives the same of wealth, self = the economy is passed as 
            ### a parameter to the agents, economy_beta = This is a economy wide 
            ### scaling parameter of how power to receive more wealth scales with 
            ### wealth itself.
            agents.append(WealthAgent(i, self.distr,
                                      self, self.economy_beta))
        return agents

    def grow(self):     
        
        ''' This method represents economic growth. One time step represents
            one day.'''

        self.time = self.time + 1
        self.help_var = self.economy_wealth
        self.economy_wealth = self.economy_wealth * (1 + self.growth_rate_economy)
        self.new_wealth = self.economy_wealth - self.help_var
          
    def choose_agent(self):
        
        ''' This method chooses an agent based on its wealth share subject to 
            the parameter beta which is the exponent/power '''
        
        weights = []
        for x in self.agents: 
            weights.append(x.wealth_share_power)
        return random.choices(self.agents, weights, k = 1)
    
    
    def sum_of_agent_power(self):
        
        ''' This method computes the sum of all agent wealth but subject to "power"-parameter
        beta which then overall gives a different sum than the normal wealth. This is 
        important so that the wealth_share_power (i.e. the wealth_share with exponent beta)
        is correctly normalized on the interval [0,1] '''
        
        sum_powers = 0
        for x in self.agents: 
           sum_powers += x.wealth**x.beta
        self.sum_power = sum_powers
    
    
    def distribute_wealth(self):
        
        ''' This method chooses an agent each around and distributes all wealth, in n-rounds, 
        where n is the number of wealth-increments (an arbitrary number that has to be chosen
         but we usually set equal to the number of agents, so that there are as
         many chances to get new wealth as as there are agents).'''
        
        for increment in range(self.increments):
            agent_j = self.choose_agent()[0]
            agent_j.wealth = agent_j.wealth + (self.new_wealth / self.increments)
            
    def recalculate_wealth_shares(self):
        for x in self.agents: 
            x.determine_wealth_share()
     
    def __repr__(self):
        return f"{self.__class__.__name__}('population size: {self.num_agents}'),('economy size: {self.economy_wealth}')"
        
        
