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



#%%


economy = Economy(10000, 100, 0.01)
### one-time procedure
economy.make_agents()


plt.hist([x.wealth for x in economy.agents])
plt.show()
data = []
for i in range(1000):
    economy.grow()
    economy.distribute_wealth()
    data.append(find_wealth_groups(economy.agents, economy.economy_wealth))
    
#top1_over_time = [x[0][0] for x in data] 
top1_share_over_time = [x[1][0] for x in data] 
top10_share_over_time = [x[1][1] for x in data] 
middle40_share_over_time = [x[1][2] for x in data] 
bottom50_share_over_time = [x[1][3] for x in data] 


plt.plot(np.linspace(1,1000,1000), top1_share_over_time)
plt.plot(np.linspace(1,1000,1000), top10_share_over_time)
plt.plot(np.linspace(1,1000,1000), middle40_share_over_time)
plt.plot(np.linspace(1,1000,1000), bottom50_share_over_time)
plt.show()
    
plt.hist([x.wealth for x in economy.agents])



