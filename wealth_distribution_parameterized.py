# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:59:25 2023

@author: earyo
"""
import os
os.chdir(".")

os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/real_time_ineq_abm")
from inequality_metrics import find_wealth_groups2
import numpy as np
from scipy.stats import powerlognorm
import matplotlib.pyplot as plt
import seaborn as sns 
#%%

### WEALTH GROUPS SHARE 01/2019 realtime-inequality.org
##https://realtimeinequality.org/
### empirical wealth shares in the UNITED STATES in January 2019 for the
### top 1%, top10%, next40% and bottom 50% of wealth owners
''' read some basic data'''

empirical_wealth_shares = [34.8, 70.9, 28.8, 0.2]
#january 2019 average wealth per adult in $
average_US_wealth_per_adult = 410400
###
###https://math.stackexchange.com/questions/2445496/weighted-sum-of-two-distributions

#%%
def PLN_normalized(c, s, sample_size):
    
    ''' samples the vector r from a powerlognorm distributuion and outputs the 
    top1%, top10%, next 40%, bottom 50%'''  
    ### This functions fits a power log-normal distribution to wealth data in the USA 
    ### Distribution and initial code taken from
    ### https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlognorm.html#scipy.stats.powerlognorm
    mean, var, skew, kurt = powerlognorm.stats(c, s, moments='mvsk')
    r = powerlognorm.rvs(c, s, size=sample_size)
    w = find_wealth_groups2(r, sum(r))[1]
    w_percent = [x*100 for x in w]
    return w_percent, r, mean


def plot_wealth_groups(bars1, bars2):
    
    ### plot wealth groups as barchart against each other  
    labels = ['top1%', 'top10%', 'next40%', 'bottom50%']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bars1, width, label='empirical')
    rects2 = ax.bar(x + width/2, bars2, width, label='PLN-model')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% of wealth in the USA')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.show()
    
    
def optimal_fit_PLN(c_range, s_range, sample_size, empirical_distr):

    ## define local variables 
    q = empirical_distr
    result_range = np.zeros((100,100))
    
    for i in range(len(c_range)):
        for j in range(len(s_range)):
            ####sample from distrbution

            r = powerlognorm.rvs(c_range[i], s_range[j], size=sample_size)
            ### find wealth groups using the find wealth groups2 fct which
            ### returns a nested list where the second element, so idx = 1, is 
            ### the wealth shares in percent
            w = find_wealth_groups2(r, sum(r))[1]
            ### express wealth groups in percentage terms
            z = [x*100 for x in w]
            
            minimization_fct = sum([abs(z[x] - q[x]) for x in range(0, 4)])
            
            result_range[i,j] = minimization_fct
            
    return result_range

    
##https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap
def heatmap2d(arr, xticks, yticks):
        plt.imshow(arr, cmap='viridis')
        plt.colorbar()
        plt.xlabel("parameter c")
        plt.ylabel("parameter s")
        plt.xticks(np.linspace(0,99,10), xticks, rotation = "vertical")  # Set label locations.
        plt.yticks(np.linspace(0,99,10), yticks)  # Set label locations.
        
        plt.show()



#%%
### plot wealth groups as barchart against each other

sampled_distr = PLN_normalized(1.05, 1.9, 10000)
groups_modelled, raw_sample, mean = sampled_distr[0], sampled_distr[1], sampled_distr[2]

plot_wealth_groups(empirical_wealth_shares, groups_modelled)
#%% 

### minimize based on absolute distance 
#### define parameter range 
c_range_data = np.around(np.linspace(0.1, 2, 100),2)
s_range_data = np.around(np.linspace(0.5, 2.5, 100),2)

#result_range_fct2 = np.zeros((100,100))

results = optimal_fit_PLN(c_range_data, s_range_data, 1000, empirical_wealth_shares)

c_range_ticks = list(c_range_data)[::10]
s_range_ticks = list(s_range_data)[::10]

heatmap2d(results, c_range_ticks, s_range_ticks)
     
