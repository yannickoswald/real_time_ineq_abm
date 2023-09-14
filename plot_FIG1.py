# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:12:26 2023

@author: earyo
"""

"""This script plots the historical indexed growth of wealth across wealth 
groups, so figure 1 of the paper. It is basically the motivation for the paper because 
one can spot substantial ups and downs in the data such as the dotcom bubble bust, the financial crisis 2008 
and the covid19 pandemic stock market bubble."""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%

with open('./data/wealth_data_for_import.csv') as f:
    d1 = pd.read_csv(f, encoding = 'unicode_escape')  

time_horizon = 46*12
colors = ["tab:red", "tab:blue", "grey", "y"]
wealth_groups = ["Top 1%", "Top 10%", "Middle 40%", "Bottom 50%"]
fig, ax = plt.subplots(figsize=(8,5))
for i, g in enumerate(wealth_groups): 
    x = d1["date_short"][d1["group"] == g].reset_index(drop = True).iloc[0:561]
    y = d1["real_wealth_growth_rate_per_unit_indexed"][d1["group"] == g].reset_index(drop = True).iloc[0:561]
    ax.plot(x,y, label = g, color = colors[i])
##define labes for plot first plot
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), frameon = False, bbox_to_anchor=(0.75, 0.5, 0.5, 0.5))
x = x.reset_index(drop=True)
ax.set_xticks(x.iloc[0::20].index)
ax.set_xticklabels(x.iloc[0::20], rotation = 90)
ax.set_ylabel("Wealth growth rate % (ref. = 01/1976)")
ax.margins(0)
ax.axvspan(280, 310, alpha=0.5, color='grey')
ax.axvspan(370, 400, alpha=0.5, color='grey')
ax.axvspan(520, 560, alpha=0.5, color='grey')
ax.annotate('Dotcom bubble', xy =(290, 5), xytext =(100, 5.1), arrowprops = dict(facecolor ='black', shrink = 0.02, linewidth=0.2))
ax.annotate('Financial crisis 07/08', xy =(380, -2), xytext =(100, -2), arrowprops = dict(facecolor ='black', shrink = 0.02, linewidth=0.2))
ax.annotate('Pandemic market shift', xy =(530, 5), xytext =(250, 3.8), arrowprops = dict(facecolor ='black',shrink = 0.02, linewidth=0.2))
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 
#%%

plt.savefig('fig1.png',  bbox_inches='tight', dpi=300)