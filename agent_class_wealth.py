# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:57:41 2023

@author: earyo
"""


#%%
class WealthAgent():
    
    '''initialization values for agent parameters'''

    def __init__(self, unique_id, wealth_begin, economy):
        
       '''
       Initialise Agent class
            
       PARAMETERS
         - unique id:             A unique agent id used for tracking individual agents
         - economy:               A class representing the economy as a whole which
                                  contains the agent also
         - wealth                 The wealth (in $) that an agent owns 
         - wealth-share           The wealth-share out of the total economy for
                                  for each agent
                         
                               
       DESCRIPTION
       Agent class that represents
       '''
       
       self.unique_id = unique_id
       self.economy = economy
       self.wealth = wealth_begin
       self.wealth_share = self.wealth / self.economy.economy_wealth
       
    
    def determine_wealth_share(self):
       self.wealth_share = self.wealth / self.economy.economy_wealth
    
    def __repr__(self):
        return f"{self.__class__.__name__}('ID: {self.unique_id}')('wealth: {self.wealth}')"
        
        

      

 
