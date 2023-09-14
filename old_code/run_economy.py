# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:58:26 2023
@author: earyo
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm  ### package for progress bars
os.chdir(".")
import matplotlib.pyplot as plt
from economy_class_wealth import Model1
from inequality_metrics import find_wealth_groups
from enkf_yo import EnsembleKalmanFilter


#%%
### LOAD empirical monthly wealth Data
with open('./data/wealth_data_for_import.csv') as f:
    d1 = pd.read_csv(f, encoding = 'unicode_escape')
    
### LOAD empirical monthly wealth Data sorted by group for state vector check
with open('./data/wealth_data_for_import2.csv') as f2:
    d2 = pd.read_csv(f2, encoding = 'unicode_escape')


#%%
# NM Naming the parameters isn't necessary but makes it easier for me to understand
economy = Model1(population_size=100, growth_rate=0.025, b_begin=1.3, distribution="Pareto_lognormal", start_year=1990)

### one-time procedure
economy.make_agents()
list_agents = economy.agents
array_agent_wealth = np.asarray([x.wealth for x in economy.agents])
plt.hist(array_agent_wealth, bins = 100)
plt.show()

time_horizon = 29*12 ## 29 years * 12 months
for i in tqdm(range(time_horizon)):
    economy.step()
    
#top1_over_time = [x[0][0] for x in data] 
top1_share_over_time = [x[1][0] for x in economy.macro_state_vectors] 
top10_share_over_time = [x[1][1] for x in economy.macro_state_vectors] 
middle40_share_over_time = [x[1][2] for x in economy.macro_state_vectors] 
bottom50_share_over_time = [x[1][3] for x in economy.macro_state_vectors] 

wealth_groups_t_data = [top1_share_over_time,
                        top10_share_over_time,
                        middle40_share_over_time,
                        bottom50_share_over_time]




#%%    
### PLOT empirical monthly wealth Data (01/1990 to 12/2018) vs model output
colors = ["tab:red", "tab:blue", "grey", "y"]
wealth_groups = ["Top 1%", "Top 10%", "Middle 40%", "Bottom 50%"]
fig, ax = plt.subplots(figsize=(6,4))
for i, g in enumerate(wealth_groups): 
    x = d1["date_short"][d1["group"] == g].reset_index(drop = True).iloc[168:516]
    y = d1["real_wealth_share"][d1["group"] == g].reset_index(drop = True).iloc[168:516]
    x1 = np.linspace(1,time_horizon,time_horizon)
    y1 = wealth_groups_t_data[i]
    ax.plot(x,np.array(y)*100, label = g, color = colors[i])
    ax.plot(x1, np.array(y1)*100, label = g + ' model', linestyle = '--', color = colors[i])
    
x = x.reset_index(drop=True)
ax.set_xticks(x.iloc[0::20].index)
ax.set_xticklabels(x.iloc[0::20], rotation = 90)
ax.legend(frameon = False, bbox_to_anchor=(0.45, 0.7, 1., .102))
ax.set_ylim((-0.05, 1))
ax.set_ylabel("Share of wealth")
ax.margins(0)
#plt.show()
plt.savefig('fig2.png',  bbox_inches='tight', dpi=300)
plt.show()
#%% plot state space of agents that is wealth vs. growth rate of wealth (in a year?)

#for s in state_vectors: 
 #   print(s[])
        
