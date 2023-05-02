# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:50:25 2023

@author: earyo
"""

import numpy as np

##### for the operator H to be a feasible transformation between micro state and
##### and macro state in our model we would need to be able to sort a vector of 
##### agent and their wealth and derive distributional metrics every time step
##### the H matrix would never be the same but vary over time itself.


#### random vector agent wealth
agent_wealth_vec = np.random.rand(10,1)*100
index_of_max = np.argmax(agent_wealth_vec)
t = np.zeros((len(agent_wealth_vec),len(agent_wealth_vec)))
t[index_of_max, index_of_max] = 1

agent_wealth_vec * t

