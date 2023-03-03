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
### LOAD empirical monthly wealth Data
with open('./data/wealth_data_for_import.csv') as f:
    d1 = pd.read_csv(f, encoding = 'unicode_escape')
#%%

economy = Economy(1000, 0.025, 1, "Pareto_lognormal", 1990)
### one-time procedure
economy.make_agents()
list_agents = economy.agents
array_agent_wealth = np.asarray([x.wealth for x in economy.agents])
plt.hist(array_agent_wealth, bins = 100)
plt.show()

data = []
time_horizon = 100
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

#%%    
### PLOT empirical monthly wealth Data (01/1990 to 12/2018)
wealth_groups = ["Top 1%", "Top 10%", "Middle 40%", "Bottom 50%"]
fig, ax = plt.subplots()
for g in wealth_groups: 
    x = d1["date_short"][d1["group"] == g].reset_index(drop = True).iloc[168:516]
    y = d1["real_wealth_share"][d1["group"] == g].reset_index(drop = True).iloc[168:516]   
    ax.plot(x,y, label = g)
x = x.reset_index(drop=True)
ax.set_xticks(x.iloc[0::20].index)
ax.set_xticklabels(x.iloc[0::20], rotation = 90)
ax.legend(frameon = False, bbox_to_anchor=(-0.7, -0.4, 1., .102))
ax.set_ylim((-0.05, 1))
ax.margins(0)