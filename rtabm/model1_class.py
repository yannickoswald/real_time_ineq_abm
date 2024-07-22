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

from exponential_pareto_avg_distr import weighted_avg_exp_pareto_distr
from exponential_pareto_avg_distr import uniform_sample


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
                 start_year,
                 uncertainty_para
                 ):
        
        ### set economy (global) attributes
        self.uncertainty_para = uncertainty_para 
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
            (i) the initial distribution of wealth which is either all exponential_pareto or
                Paretolognormal
            (ii) the year that the model starts'''
        
        ## intialize list of agents
        agents = list()

        if self.distr == "Pareto_lognormal":
        
            for i in range(self.num_agents):
                ### The agent parameters are: i = unique_id, self.distr, self = the economy is passed as 
                ### a parameter to the agents, economy_beta = This is a economy wide 
                ### scaling parameter of how power to receive more wealth scales with 
                ### wealth itself.
                ### set distribution from which agents' wealth is sampled initially
                if self.distr == "Pareto_lognormal":
                    ### the scaling_coefficient is determined by fitting the wealth average
                    ### of the sample distr. to the empirical wealth average
                    #if self.start_year == 2019:
                    scale_coeff =  200000
                    a_wealth = powerlognorm.rvs(0.33, 1.15, size=1)*scale_coeff
                    #elif self.start_year == 1990:
                     #   scale_coeff = 150000
                      #  a_wealth = powerlognorm.rvs(1.92, 2.08, size=1)*scale_coeff
            
                ## introduce variable q only for clarity
                q = self.economy_beta # + np.random.normal(0, 0.1*self.economy_beta)
                ## create agent
                agents.append(Agent1(i, a_wealth, self, q))

        elif self.distr == "exponential_pareto":


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
            average_wealth = d_average[d_average["Year"] == self.start_year]["Real Wealth Per Unit"].values[0]
            # find scale factor for the year the model starts per formula derived in test calibration new weighted avg
            scale_factor = 0.04*average_wealth ## applied equation from test_calibration_new_weighted_avg
            #print(f"average wealth: {average_wealth}, scale factor: {scale_factor}")

            sample = uniform_sample(self.num_agents)
            sample_of_agents = weighted_avg_exp_pareto_distr(sample, 0.4, 0.9, alpha = 1.3, Temperature = 5)
            for i in range(self.num_agents):
                a_wealth = sample_of_agents[i]*scale_factor
                q = self.economy_beta # + np.random.normal(0, 0.1*self.economy_beta)
                agents.append(Agent1(i, a_wealth, self, q))

        else:
            raise Exception(f"Unrecognised distribution: {self.distr}")
        
        return agents

    def grow(self):
        ''' This method represents economic growth. One time step represents
            one month.'''

        help_var = self.economy_wealth
        self.economy_wealth = self.economy_wealth * (1 + self.growth_rate_economy)
        self.new_wealth = self.economy_wealth - help_var
          
    def choose_agent(self):

        #This method chooses an agent based on its wealth share subject to 
         #   the parameter beta which is the exponent/power 
        
        weights = [x.wealth_share_power for x in self.agents]
        #print(f"sum of weights {sum(weights)}")
        self.weightshistory.append(weights)
        return random.choices(self.agents, weights, k=1)[0]
    

    def sum_of_agent_power(self):   

        ''' This method computes the sum of all agent wealth but subject to a "power"-parameter
        beta which then overall gives a different sum than the normal wealth. This is 
        important so that the wealth_share_power (i.e. the wealth_share with exponent beta)
        is correctly normalized on the interval [0,1]. Negative wealth agents are ignored because
        they anyway will be assigned a weight of 0 and cannot receive more wealth.'''

        return sum([x.wealth**x.beta for x in self.agents if x.wealth > 0])

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

        print("This is the macro state vectors ", self.macro_state_vectors[0][0])

        if len(self.macro_state_vectors[0][0]) == 4:

            print("This is the macro state vectors length", len(self.macro_state_vectors))
        
            top1_share_over_time = [x[1][0] for x in self.macro_state_vectors] 
            top10_share_over_time = [x[1][1] for x in self.macro_state_vectors] 
            middle40_share_over_time = [x[1][2] for x in self.macro_state_vectors] 
            bottom50_share_over_time = [x[1][3] for x in self.macro_state_vectors] 

            return [top1_share_over_time,
                    top10_share_over_time,
                    middle40_share_over_time,
                    bottom50_share_over_time]
        
        elif len(self.macro_state_vectors[0][0]) == 3:

            print("This is the macro state vectors length", len(self.macro_state_vectors))
             
            top10_share_over_time = [x[1][0] for x in self.macro_state_vectors] 
            middle40_share_over_time = [x[1][1] for x in self.macro_state_vectors] 
            bottom50_share_over_time = [x[1][2] for x in self.macro_state_vectors] 

            return [top10_share_over_time,
                    middle40_share_over_time,
                    bottom50_share_over_time]
        
    def plot_wealth_groups_over_time(self, ax, start_year, end_year):
        
        ''' PLOT empirical monthly wealth Data specified period vs model output
        for period + 1 month'''

        ### LOAD empirical monthly wealth Data
        path = ".."
        with open(os.path.join(path, 'data', 'wealth_data_for_import2.csv')) as f:
            d1 = pd.read_csv(f, encoding = 'unicode_escape')
            
        wealth_groups_t_data = self.collect_wealth_data()
        print("This is the wealth groups data", wealth_groups_t_data)
        ### PLOT empirical monthly wealth Data (01/1990 to 12/2018) vs model output
        colors = ["tab:red", "tab:blue", "grey", "y"]
        if len(wealth_groups_t_data) == 3:
            wealth_groups = ["Top 10%", "Middle 40%", "Bottom 50%"]
        elif len(wealth_groups_t_data) == 4:
            wealth_groups = ["Top 1%", "Top 10%-1%", "Middle 40%", "Bottom 50%"]
        # use start and end year to determine the period length end year +1 because of python indexing and wanting to include last year
        period_length_years = (end_year+1) - start_year 
        period_length_months = period_length_years * 12 # all months

        # compute the points how to subset the data
        first_year_available = 1976
        start_point = (start_year - first_year_available)*12
        end_point = ((end_year+1) - first_year_available)*12

        for i, g in enumerate(wealth_groups): 
            x = d1["date_short"][d1["group"] == g].reset_index(drop = True).iloc[start_point:end_point]
            y = d1["real_wealth_share"][d1["group"] == g].reset_index(drop = True).iloc[start_point:end_point]
            x1 = np.linspace(1,  period_length_months, period_length_months)
            y1 = wealth_groups_t_data[i]
            ax.plot(x,y, label = g, color = colors[i], linestyle = '--')
            ax.plot(x1, y1, label = g + ' model', linestyle = '-', color = colors[i])
             
        x = x.reset_index(drop=True)
        ax.set_xticks(x.iloc[0::20].index)
        ax.set_xticklabels(x.iloc[0::20], rotation = 90)
        #ax1.legend(frameon = False, bbox_to_anchor=(0.45, 0.7, 1., .102))
        ax.set_ylim((-0.05, 0.8))
        ax.set_yticklabels(['0%', '0%', '20%', '40%', '60%', '80%'])
        #ax.set_yticklabels(['0%', '0%', '10%', '20%', '30%', '40%', '50%', '60%'])
        ax.set_ylabel("wealth share")
        ax.margins(0)
        #ax.text(0,0.85, 'a',  fontsize = 12)
        
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
