# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:58:26 2023

@author: earyo
"""

import os
import pandas as pd
import sys
import numpy as np
import random
os.chdir(".")

import matplotlib.pyplot as plt
from economy_class_wealth import Economy
from inequality_metrics import find_wealth_groups

#### implement progress bars ####
#%%

economy = Economy(100000, 100, 0.025, 1)
### one-time procedure
economy.make_agents()
list_agents = economy.agents

plt.hist([x.wealth for x in economy.agents])
plt.show()


data = []
time_horizon = 10000
for i in range(time_horizon):
    economy.sum_of_agent_power()
    economy.grow()
    economy.distribute_wealth()
    data.append(find_wealth_groups(economy.agents, economy.economy_wealth))
    economy.recalculate_wealth_shares()
    
#top1_over_time = [x[0][0] for x in data] 
top1_share_over_time = [x[1][0] for x in data] 
top10_share_over_time = [x[1][1] for x in data] 
middle40_share_over_time = [x[1][2] for x in data] 
bottom50_share_over_time = [x[1][3] for x in data] 


plt.plot(np.linspace(1,time_horizon,time_horizon), top1_share_over_time, label = "top1%")
plt.plot(np.linspace(1,time_horizon,time_horizon), top10_share_over_time, label = "top10%")
plt.plot(np.linspace(1,time_horizon,time_horizon), middle40_share_over_time, label = "middle 40%")
plt.plot(np.linspace(1,time_horizon,time_horizon), bottom50_share_over_time, label = "bottom 50%")
plt.legend()
plt.show()
    
plt.hist([x.wealth for x in economy.agents])
