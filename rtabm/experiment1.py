# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:20:14 2023

@author: earyo
"""

#General packages
import os
import numpy as np
from tqdm import tqdm  ### package for progress bars
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
### MODEL 1 infrastructure
import pandas as pd
from model1_class import Model1
from run_enkf import *
### MODEL 2 infrastructure
from model2_class import Model2
from run_enkf2 import *
#from run_both_models_n_times_and_compute_error import *


#%%

#if __name__=="__main__":
enkf1 = prepare_enkf(num_agents=100, ensemble_size=30, macro_state_dim=4)
enkf2 = prepare_enkf2(num_agents=100, ensemble_size= 30, macro_state_dim = 4)

run_enkf(enkf1)
run_enkf(enkf2)

# Now let's say you want to integrate this into another grid layout
fig = plt.figure(figsize=(10, 10))
# Create a gridspec object
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
# Create individual subplots
ax0 = plt.subplot(gs[0, 0])
ax1 = plt.subplot(gs[0, 1])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[1, 1])
ax4 = plt.subplot(gs[2, :])  # This one spans both columns

enkf1.models[0].plot_wealth_groups_over_time(ax0, 29*12)
enkf2.models[0].plot_wealth_groups_over_time(ax1, 29*12)
enkf1.plot_fanchart(ax2)
enkf2.plot_fanchart(ax3)
enkf1.plot_error(ax4)
enkf2.plot_error(ax4)

###EXTRAS
#AX0
ax0.text(0, 0.85, 'a', fontsize = 12)
ax0.text(40, 0.85, 'Model 1', fontsize = 12)
#AX1
ax1.legend(loc=(1.05, -0.15), frameon = False) ### legend only here
ax1.text(0, 0.85, 'b', fontsize = 12)
ax1.text(40, 0.85, 'Model 2', fontsize = 12)
#AX2
ax2.text(0, 1.05, 'c', fontsize = 12)
ax2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax2.text(40,1.05, 'Model 1', fontsize = 12)
#AX3
ax3.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax3.text(0,1.05, 'd', fontsize = 12)
ax3.text(40,1.05, 'Model 2', fontsize = 12)
# Get the limits
x_min, x_max = ax4.get_xlim()
y_min, y_max = ax4.get_ylim()
ax4.text(0,y_max+0.02, 'e', fontsize = 12)
ax4.legend(frameon= False)
plt.tight_layout()
#plt.savefig('fig2.png', dpi = 300)

plt.show()
