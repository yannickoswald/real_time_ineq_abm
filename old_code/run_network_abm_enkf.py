# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:43:56 2023

@author: earyo
"""
from rtabm.economy_class_wealth2 import *
from rtabm.enkf_yo2 import EnsembleKalmanFilter

model = Model(500, concavity=0.01, growth_rate = 0.02, start_year = 1990)  # 100 agents
for _ in range(360):  # Run for 10 steps
    model.step()
model.plot_network()
model.plot_wealth_histogram()
model.plot_wealth_groups_over_time()



#%%


num_agents = 100
filter_params = {"ensemble_size": 10,
                 "macro_state_vector_length": 4,
                 "micro_state_vector_length": num_agents}

model_params = {"num_agents": num_agents,
 "growth_rate": 0.025,
 "concavity": 0.1,
 "start_year": 1990 }

model_params2 = {"num_agents": num_agents,
                 "growth_rate": 0.025,
                 "beta": 1.3,
                 "distribution": "Pareto_lognormal",
                 "start_year": 1990}


enkf = EnsembleKalmanFilter(Model, filter_params, model_params = model_params)
enkf2 = EnsembleKalmanFilter(Economy, filter_params, model_params = model_params2)


enkf.plot_fanchart()