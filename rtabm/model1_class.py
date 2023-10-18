# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 09:46:44 2023

@author: earyo
"""

### import necessary libraries
import os
import numpy as np
import random
from agent1_class import Agent1
from scipy.stats import powerlognorm
from inequality_metrics import find_wealth_groups
import pandas as pd

#%%

class Model1():
    
    """A model of the wealth distribution in the economy. 
       Wealth is defined as the physical and financial assets someone
       (or society as a whole) holds and is distinct from the income 
       or the consumption of a person."""
       
    def __init__(self, 
                 population_size,
                 growth_rate,
                 b_begin,
                 distribution: str,
                 start_year
                 ):
        
        ### set economy (global) attributes
        self.start_year = start_year
        self.num_agents = population_size  
        self.growth_rate_economy = (1+growth_rate)**(1/12) - 1## ~growth_rate / 12 ### MONTHLY growth rate
        #### the number of increments is important since it determines how the new
        #### wealth growth is divided and how many chances there are to receive some. 
        self.increments = 100
        ### SET OTHER MODEL PARAMETERS
        ### track time in the economy 
        self.time = 0    
        ### Beta is an economy wide 
        ### scaling parameter of how power to receive more wealth scales with 
        ### wealth itself.
        self.economy_beta = b_begin
        ### distribution type in the economy that 
        ### determines the initial distribution of wealth
        self.distr = distribution
        ### set economy agents
        self.agents = self.make_agents()
        ### compute total economy wealth bottom up from the level of agents
        total_wealth = sum([x.wealth for x in self.agents])
        self.economy_wealth = total_wealth
        ### The sum_power is a model parameter which helps calculate the 
        ### normalized wealth share of
        ### an agent given the agent parameter beta. Initialized as the usual 
        ### sum of wealth
        self.sum_power = total_wealth
        ### state-space and data storing 
        ### the macro state = AVERAGE WEALTH PER ADULT PER GROUP
        ### the micro state = WEALTH PER EACH AGENT
        self.macro_state = None
        self.micro_state = None
        self.macro_state_vectors = [] ### wealth group data 
        self.micro_state_vectors = [] ### system state on agent level
        self.new_wealth = None
        self.weightshistory = []

        
    def make_agents(self):
        ''' method makes agents dependent on:
            (i) the initial distribution of wealth which is either all equal or
                Paretolognormal
            (ii) the year that the model starts'''
        agents = list()
        for i in range(self.num_agents):
            ### The agent parameters are: i = unique_id, self.distr, self = the economy is passed as 
            ### a parameter to the agents, economy_beta = This is a economy wide 
            ### scaling parameter of how power to receive more wealth scales with 
            ### wealth itself.
            ### set distribution from which agents' wealth is sampled initially
            if self.distr == "all_equal":
                a_wealth = 10000
            elif self.distr == "Pareto_lognormal":
                ### the scaling_coefficient is determined by fitting the wealth average
                ### of the sample distr. to the empirical wealth average
                if self.start_year == 2019:
                    scale_coeff =  200000
                    a_wealth = powerlognorm.rvs(0.33, 1.15, size=1)*scale_coeff
                elif self.start_year == 1990:
                    scale_coeff = 150000
                    a_wealth = powerlognorm.rvs(1.92, 2.08, size=1)*scale_coeff
            else:
                raise Exception(f"Unrecognised distribution: {self.distr}")
            ## introduce variable q only for aesthetic purpose
            q = self.economy_beta
             ## create agent
            agents.append(Agent1(i, a_wealth, self, q))
        return agents

    def grow(self):
        ''' This method represents economic growth. One time step represents
            one month.'''

        help_var = self.economy_wealth
        self.economy_wealth = self.economy_wealth * (1 + self.growth_rate_economy)
        self.new_wealth = self.economy_wealth - help_var
          
    def choose_agent(self):
        ''' This method chooses an agent based on its wealth share subject to 
            the parameter beta which is the exponent/power '''  
        weights = [x.wealth_share_power for x in self.agents]
        self.weightshistory.append(weights)
        return random.choices(self.agents, weights, k=1)[0]

    def sum_of_agent_power(self):   
        ''' This method computes the sum of all agent wealth but subject to a "power"-parameter
        beta which then overall gives a different sum than the normal wealth. This is 
        important so that the wealth_share_power (i.e. the wealth_share with exponent beta)
        is correctly normalized on the interval [0,1] '''
        return sum([x.wealth**x.beta for x in self.agents])

    def distribute_wealth(self):
        ''' This method chooses an agent each around and distributes all wealth, in n-rounds, 
        where n is the number of wealth-increments (an arbitrary number that has to be chosen
         but we usually set equal to the number of agents, so that there are as
         many chances to get new wealth as as there are agents).'''     
        for increment in range(self.increments):
            agent_j = self.choose_agent()
            agent_j.wealth = agent_j.wealth + (self.new_wealth / self.increments)
            
    def recalculate_wealth_shares(self):
        for x in self.agents: 
            x.determine_wealth_share()
                
    def determine_agent_trajectories(self):
        for x in self.agents: 
            x.det_wealth_trajectory()
    
    def micro_state_vec_data(self):
        ## two rows because we have w = wealth and wealth change rate as critical variables
        sv_data = np.zeros((self.num_agents, 2))   
        for count, x in enumerate(self.agents): 
            sv_data[count,0] = x.wealth
            sv_data[count,1] = x.g_rate
        return sv_data
    
    def step(self): 
        self.time = self.time + 1
        self.sum_power = self.sum_of_agent_power()
        self.grow()
        self.distribute_wealth()
        self.determine_agent_trajectories()
        a = find_wealth_groups(self.agents, self.economy_wealth)
        self.macro_state_vectors.append(a)
        self.macro_state = np.array(a[0])
        self.micro_state_vectors.append((self.micro_state_vec_data()))
        self.micro_state = self.micro_state_vec_data()[:,0]
        self.recalculate_wealth_shares()
        
    def update_agent_states(self):
        '''update agent states after EnKF state vector update. Needs 
        possibly to be verified further to ensure correct running results.'''
        for count, x in enumerate(self.agents): 
            x.wealth = self.micro_state[count]
            #assert x.wealth > 0
            #for count, x in enumerate(enkf.models[0].agents): print(count, x, x.wealth,enkf.models[0].micro_state[count])
    
    def collect_wealth_data(self):
        
        ''' Collects macro wealth data for plotting and analysis.'''
        
        top1_share_over_time = [x[1][0] for x in self.macro_state_vectors] 
        top10_share_over_time = [x[1][1] for x in self.macro_state_vectors] 
        middle40_share_over_time = [x[1][2] for x in self.macro_state_vectors] 
        bottom50_share_over_time = [x[1][3] for x in self.macro_state_vectors] 

        return [top1_share_over_time,
                top10_share_over_time,
                middle40_share_over_time,
                bottom50_share_over_time]
        
    def plot_wealth_groups_over_time(self, ax, period):
        
        ''' PLOT empirical monthly wealth Data specified period vs model output'''
        ### LOAD empirical monthly wealth Data
        path = ".."
        with open(os.path.join(path, 'data', 'wealth_data_for_import.csv')) as f:
            d1 = pd.read_csv(f, encoding = 'unicode_escape')
            
        wealth_groups_t_data = self.collect_wealth_data()
        ### PLOT empirical monthly wealth Data (01/1990 to 12/2018) vs model output
        colors = ["tab:red", "tab:blue", "grey", "y"]
        wealth_groups = ["Top 1%", "Top 10%-1%", "Middle 40%", "Bottom 50%"]
        for i, g in enumerate(wealth_groups): 
            x = d1["date_short"][d1["group"] == g].reset_index(drop = True).iloc[168:516]
            y = d1["real_wealth_share"][d1["group"] == g].reset_index(drop = True).iloc[168:516]
            x1 = np.linspace(1,period,period)
            y1 = wealth_groups_t_data[i]
            ax.plot(x,y, label = g, color = colors[i], linestyle = '--')
            ax.plot(x1, y1, label = g + ' model', linestyle = '-', color = colors[i])
            
        x = x.reset_index(drop=True)
        ax.set_xticks(x.iloc[0::20].index)
        ax.set_xticklabels(x.iloc[0::20], rotation = 90)
        #ax1.legend(frameon = False, bbox_to_anchor=(0.45, 0.7, 1., .102))
        ax.set_ylim((-0.05, 0.61))
        ax.set_yticklabels(['0%', '0%', '10%', '20%', '30%', '40%', '50%', '60%'])
        ax.set_ylabel("wealth share")
        ax.margins(0)
        ax.text(0,0.65, 'a',  fontsize = 12)
        
        ### convert wealth group data to array
        result_array = np.column_stack([np.array(lst) for lst in wealth_groups_t_data])
        return result_array
        
    def write_data_for_plots(self):
        
        """
        Collect data for ensemble plots if not used in ENKF
        """
        return self.collect_wealth_data()

    def __repr__(self):
        return f"{self.__class__.__name__}('population size: {self.num_agents}'),('economy size: {self.economy_wealth}')"
