# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:57:41 2023

@author: earyo
"""


#%%
class WealthAgent():
    
    '''initialization values for agent parameters'''

    def __init__(self, unique_id, wealth_begin, economy, wealth_exponent):
        
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
       self.wealth_share = 1#self.wealth / self.economy.economy_wealth
       self.wealth_share_power = 1#self.wealth / self.economy.economy_wealth
       self.beta = wealth_exponent
       
    
    def determine_wealth_share(self):
       ### actual wealth share
       self.wealth_share = self.wealth / self.economy.economy_wealth
       ### the power that the wealth-share provides so to speak in order to gain new wealth
       ### is a function of beta
       self.wealth_share_power = (self.wealth**self.beta) / self.economy.sum_power
       
       
    def __repr__(self):
        return f"{self.__class__.__name__}('ID: {self.unique_id}')('wealth: {self.wealth}')"
        
        

      

 
