# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:53:37 2023

@author: earyo
"""

def find_wealth_groups(agents_list, total_wealth):
    
    '''receives a list of agents, sorts them and gives back the absolute wealth,
    and the wealth share, of the top1%, 10%, next 40%, bottom 50%, and bottom 10% '''

    ## sort agent list from rich (at top of list) to poor (at bottom of list)
    agents_list.sort(key=lambda x: x.wealth, reverse=True)
    
    '''compute various range parameters for wealth groups'''
    
    ## compute number of agents in one percentile 
    agents_per_percentile = int(len(agents_list)/100)
    
    ## number of agents in top 1%
    agents_per_top1 = agents_per_percentile
    ## number of agents in top 10%
    agents_per_top10 = agents_per_percentile*10
    ## number of agents in next 40%
    agents_per_next40 = agents_per_percentile*40
    ## number of agents in bottom 50%
    agents_per_bottom50 = agents_per_percentile*50
    ##### required to compute next 40%
    ## number of agents in top 30% and bottom 30%
    agents_per_top30 = agents_per_percentile*30
    
    ### LISTS of WEALTH GROUPS
    top1 = agents_list[:agents_per_top1]
    top10 = agents_list[:agents_per_top10]
    next40 = agents_list[agents_per_top30:len(agents_list)-agents_per_top30]
    bottom50 = agents_list[agents_per_bottom50:]
    
    ### LOOP OVER ALL LISTS of WEALTH GROUPS AND SUM WEALTH OF AGENTS
    ###initialize values
    top1_wealth = 0
    top10_wealth = 0
    next40_wealth = 0
    bottom50_wealth = 0
    
    ##cannot be done in one loop since lists are of distinct lengths
    ## could be done with itertools, however then a fill object is required which
    ## overcomplicates things
    
    for a in top1:   
        top1_wealth += a.wealth
    for b in top10:
        top10_wealth += b.wealth
    for c in next40:
        next40_wealth += c.wealth
    for d in bottom50:
        bottom50_wealth += d.wealth
        
    ### compute wealth shares
    share_top1 = top1_wealth / total_wealth
    share_top10 = top10_wealth / total_wealth
    share_next40 = next40_wealth / total_wealth
    share_bottom50 = bottom50_wealth / total_wealth
    
    
    return [[top1_wealth, top10_wealth,  next40_wealth, bottom50_wealth]
            ,[share_top1, share_top10, share_next40, share_bottom50]]
    

def find_wealth_groups2(values, total_wealth):
    
    '''receives an array and sorts and gives back the absolute wealth,
    and the wealth share, of the top1%, 10%, next 40%, bottom 50%, and bottom 10% '''

    ## sort agent list from rich (at top of list) to poor (at bottom of list)
    values[::-1].sort()
    
    '''compute various range parameters for wealth groups'''
    
    ## compute number of agents in one percentile 
    agents_per_percentile = int(len(values)/100)
    
    ## number of agents in top 1%
    agents_per_top1 = agents_per_percentile
    ## number of agents in top 10%
    agents_per_top10 = agents_per_percentile*10
    ## number of agents in next 40%
    agents_per_next40 = agents_per_percentile*40
    ## number of agents in bottom 50%
    agents_per_bottom50 = agents_per_percentile*50
    ##### required to compute next 40%
    ## number of agents in top 30% and bottom 30%
    agents_per_top30 = agents_per_percentile*30
    
    ### LISTS of WEALTH GROUPS
    top1 = values[:agents_per_top1]
    top10 = values[:agents_per_top10]
    next40 = values[agents_per_top10:len(values)-agents_per_bottom50]
    bottom50 = values[agents_per_bottom50:]
    
    ### LOOP OVER ALL LISTS of WEALTH GROUPS AND SUM WEALTH OF AGENTS
    ###initialize values
    top1_wealth = 0
    top10_wealth = 0
    next40_wealth = 0
    bottom50_wealth = 0
    
    ##cannot be done in one loop since lists are of distinct lengths
    ## could be done with itertools, however then a fill object is required which
    ## overcomplicates things
    
    for a in top1:   
        top1_wealth += a
    for b in top10:
        top10_wealth += b
    for c in next40:
        next40_wealth += c
    for d in bottom50:
        bottom50_wealth += d
        
    ### compute wealth shares
    share_top1 = top1_wealth / total_wealth
    share_top10 = top10_wealth / total_wealth
    share_next40 = next40_wealth / total_wealth
    share_bottom50 = bottom50_wealth / total_wealth
    
    
    return [[top1_wealth, top10_wealth,  next40_wealth, bottom50_wealth]
            ,[share_top1, share_top10, share_next40, share_bottom50]]
    
