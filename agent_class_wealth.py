# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:57:41 2023

@author: earyo
"""

import numpy as np

#%%
class WealthAgent():
    
    '''initialization values for agent parameters'''

    def __init__(self, unique_id: int, wealth_begin: float, economy, wealth_exponent: float):
        
       '''
       Initialise Agent class
            
       PARAMETERS
         - unique id:             A unique agent id used for tracking individual agents
         - economy:               A class representing the economy as a whole which
                                  contains the agent also
         - wealth                 The wealth (in $) that an agent owns 
         - wealth_share           The wealth-share out of the total economy for
                                  for each agent
         - wealth_share_power     The computed power/weight an agent receives 
                                  which is basically the probability that they receive
                                  another increment of wealth.
         - beta/wealth-exponent   β (beta) is an exponent that determines whether the wealth share
                                  vs. probability to receive more wealth is a concave 
                                  or convex relationship (concave = saturating, convex = escalating). When 
                                  β = 1, then relationship is just proportional. In other words, it
                                  is a economy wide scaling parameter of 
                                  how power to receive more wealth scales with 
                                  wealth itself.
                                           
       DESCRIPTION
       Agent class that represents one person in the American economy. 
       Although that does not mean we handle a realistic number of agents. We might 
       work on the order of thousands or even hundreds instead of hundred of millions.
       '''
       
       self.unique_id = unique_id
       self.economy = economy
       self.wealth = wealth_begin
       self.wealth_share = 1 #placeholder value
       self.wealth_share_power = 1 #placeholder value
       self.beta = wealth_exponent
       self.wealth_growth_rate = 0
       self.wealth_list = []
       self.g_rate = 0 ### growth rate of wealth for individual agent
       self.g_rate_list = []
       self.wealth_variance = 0.1*self.g_rate
       self.g_rate_variance =  0.1*self.g_rate
    
    def det_wealth_trajectory(self):
        self.wealth_list.append(self.wealth)
        if len(self.wealth_list) >= 2:
            self.g_rate = (self.wealth_list[-1] / self.wealth_list[-2]) - 1
        self.g_rate_list.append(self.g_rate)

    def determine_wealth_share(self):
       ### actual wealth share
       #self.wealth_share = self.wealth / self.economy.economy_wealth
       ### the power that the wealth-share provides so to speak in order to gain new wealth
       ### is a function of beta
      
       ### if the agent state is updated it can happen that the agent wealth is less than 0, 
       ### control for this otherwise the agent wealth weights do not make much sense
       ### for the selection of an agent to distribute more wealth
       if self.wealth > 0: 
           z = ((self.wealth**self.beta) / self.economy.sum_power)
       else: 
           z = 0.8
           
       ### also truncate distribution of it at 0
       self.wealth_share_power = max(0, z) 
    

    def __repr__(self):
        return f"{self.__class__.__name__}('ID: {self.unique_id}')('wealth: {self.wealth}')"
        
        

      

 
