# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:59:25 2023

@author: earyo

In this script we fit a Pareto-lognormal distribution to the wealth distribution
data in the USA from https://realtimeinequality.org/
"""
import os
os.chdir(".")
#os.chdir("C:/Users/earyo/Dropbox/Arbeit/postdoc_leeds/real_time_ineq_abm")
from inequality_metrics import find_wealth_groups2
import numpy as np
from scipy.stats import powerlognorm
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm import tqdm
#%%
### WEALTH GROUPS SHARE 01/2019 realtime-inequality.org
''' read some basic data'''
'''2019'''
### empirical wealth shares in the UNITED STATES in January 2019 for the
### top 1%, top10%, next40% and bottom 50% of wealth owners
empirical_wealth_shares = [34.8, 70.9, 28.8, 0.2]
#january 2019 average wealth per adult in $
average_US_wealth_per_adult = 410400
'''1990'''
### 1990 data is important so we can validate the model by 
### fitting it to the data from 1990 to 2019
####empirical wealth_shares for january 1990 [28.6, 64.7, 33.4, 1.8]
empirical_wealth_shares_1990 = [28.6, 64.7, 33.4, 1.8]
#january 1990 average wealth per adult in $
average_US_wealth_per_adult_1990 = 203900

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
    result_range = np.zeros((len(c_range),len(s_range)))
    for i in tqdm(range(len(c_range))):
        for j in range(len(s_range)):
            ####sample from distrbution
            r = powerlognorm.rvs(c_range[i], s_range[j], size=sample_size)
            ### find wealth groups using the find wealth groups2 fct which
            ### returns a nested list where the second element, so idx = 1, is 
            ### the wealth shares in percent
            w = find_wealth_groups2(r, sum(r))[1]
            ### express wealth groups in percentage terms
            z = [x*100 for x in w]
            #print(z)
            #print(q)
            ### minimize based on absolute distance 
            minimization_fct = sum([abs(z[x] - q[x]) for x in range(0, 4)])
            #print(minimization_fct)
            #print(c_range[i],s_range[j], minimization_fct)
            result_range[i,j] = minimization_fct    
    return result_range

##https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap
def heatmap2d(arr, xticks, yticks):
        plt.imshow(arr, cmap='viridis')
        plt.title('Minimize abs error of percentages between wealth groups')
        plt.colorbar()
        plt.xlabel("parameter s")
        plt.ylabel("parameter c")
        plt.xticks(np.linspace(0,99,10), yticks, rotation = "vertical")  # Set label locations.
        plt.yticks(np.linspace(0,99,10), xticks)  # Set label locations.        
        plt.show()


#%% 
#### define parameter range  that is to be searched over for an optimum/min.
c_range_data = np.around(np.linspace(0.1, 2, 100),2)
s_range_data = np.around(np.linspace(0.5, 2.5, 100),2)
sample_size = 10**5
#%% RUN ERROR MINIMIZATION for 2019 (January)
### plot wealth groups as barchart against each other
sampled_distr = PLN_normalized(0.33, 1.15, sample_size)
groups_modelled, raw_sample, mean = sampled_distr[0], sampled_distr[1], sampled_distr[2]
plot_wealth_groups(empirical_wealth_shares, groups_modelled)
### minimize based on absolute distance 
#result_range_fct2 = np.zeros((100,100))
results = optimal_fit_PLN(c_range_data, s_range_data, sample_size, empirical_wealth_shares)
results[np.where(c_range_data==1.04) , np.where(s_range_data==1.95)]
#np.where(results == np.min(results))
#results[np.where(results == np.min(results))[0][0], np.where(results == np.min(results))[1][0]]
optimal_c = c_range_data[np.where(results == np.min(results))[0][0]]   
optimal_s = s_range_data[np.where(results == np.min(results))[1][0]]
c_range_ticks = list(c_range_data)[::10]
s_range_ticks = list(s_range_data)[::10]
heatmap2d(results, c_range_ticks, s_range_ticks)

#%% RUN ERROR MINIMIZATION for 1990 January
results_1990 = optimal_fit_PLN(c_range_data, s_range_data, sample_size, empirical_wealth_shares_1990)
results_1990[np.where(c_range_data==1.04) , np.where(s_range_data==1.95)]
#np.where(results == np.min(results))
#results[np.where(results == np.min(results))[0][0], np.where(results == np.min(results))[1][0]]
optimal_c_1990 = c_range_data[np.where(results_1990 == np.min(results_1990))[0][0]]   
optimal_s_1990 = s_range_data[np.where(results_1990 == np.min(results_1990))[1][0]]