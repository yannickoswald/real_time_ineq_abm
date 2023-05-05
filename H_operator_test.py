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
agent_wealth_vec = np.linspace(1,10,10).reshape((10, 1))
wealth_group_data = np.linspace(1,4,4).reshape((4, 1))


H = np.zeros((10,4))
H.shape
agent_wealth_vec.shape

H[:int(10*0.01),0] = 1
H[:int(10*0.1),1] = 1
H[int(10*0.1):int(10*0.5),2] = 1
H[int(10*0.5):,3] = 1

micro_to_macro_test = agent_wealth_vec.T@H


def make_H(dim_micro_state, dim_data):
    H = np.zeros((dim_micro_state, dim_data))
    H[:int(10*0.01),0] = 1
    H[:int(10*0.1),1] = 1
    H[int(10*0.1):int(10*0.5),2] = 1
    H[int(10*0.5):,3] = 1
    return H
    