# -*- coding: utf-8 -*-
"""
@author: Yannick Oswald while @University of Leeds, School of Geography 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import random as random
from inequality_metrics import find_wealth_groups2
import numpy as np
from scipy.stats import powerlognorm
import pandas as pd
from agent2_class import Agent2


class Model2:
    
    ''' Implements a network-based agent-based model '''
    
    def __init__(self, num_agents, concavity, growth_rate, start_year, adaptive_sensitivity):
        self.num_agents = num_agents
        self.agents = [Agent2(i, self, adaptive_sensitivity) for i in range(num_agents)]
        self.graph = self.create_network()
        self.growth_rate = growth_rate
        self.wealth_data = list()
        self.concavity = concavity
        self.start_year = start_year ## doesn't do anything currently
        self.time = 0
        self.macro_state = None
        self.micro_state = None

    def create_network(self):
        """Create a graph with Barabasi-Albert degree distribution and place
        agent on this graph"""
        # Generate a network with m (desired number of edges each new node should have) = 2
        G = nx.barabasi_albert_graph(self.num_agents, 2)
        ### sort agents by wealt
        sorted_agents = sorted(self.agents, key=lambda agent: agent.wealth, reverse=True)
        indices_of_sorted_agents = [a.id for a in sorted_agents]
        ### sort nodes by node degree
        sorted_nodes = sorted(G.nodes, key=lambda node: G.degree[node], reverse=True)
        # Assign sorted agents to sorted nodes
        for i in range(len(self.agents)):
             G.nodes[sorted_nodes[i]]['agent'] = self.agents[indices_of_sorted_agents[i]]
        return G    
    
    def get_wealth_data(self):
        """Collect wealth data over time as plot input"""
        return [a.wealth for a in self.agents]
         
    def step(self):
        """Advance the model by one step"""
        # Randomly order agents and let them act in that order
        random_order = random.sample(self.agents, len(self.agents))
        for agent in random_order:
            agent.step(self)
            
        # Collect wealth data
        self.wealth_data.append(self.get_wealth_data())
        self.time = self.time + 1

    def plot_network(self):
        """Plot the network of agents"""
        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, with_labels=True,
                node_color='skyblue', node_size=1500,
                edge_cmap=plt.cm.Blues)
        plt.show()
        
    def plot_wealth_histogram(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,10))
        wealth_data = [a.wealth for a in self.agents]
        ax.hist(wealth_data)
        ax.set_xlabel("wealth")
        ax.set_ylabel("frequency")
        
    def plot_wealth_groups_over_time(self, ax):
        """
        Plot data on the given axes.
        """
        ### LOAD empirical monthly wealth Data

        with open('./data/wealth_data_for_import.csv') as f:
            d1 = pd.read_csv(f, encoding = 'unicode_escape')  
        
            
        colors = ["tab:red", "tab:blue", "grey", "y"]
        wealth_groups = ["Top 1%", "Top 10%", "Middle 40%", "Bottom 50%"]
        groups_over_time = list()
        for i in range(0, len(self.wealth_data)):
            a = np.array(self.wealth_data[i])
            b = sum(self.wealth_data[i])
            t = find_wealth_groups2(a, b)[1]
            groups_over_time.append(t)
        y = np.vstack(groups_over_time)
        #y =  ### select the correct times
        x = np.linspace(1, len(y), len(y))
        
       
        for i, g in enumerate(wealth_groups):    
            key = d1["group"] == wealth_groups[i]
            y2 = d1["real_wealth_share"][key].reset_index(drop = True).iloc[168:516]
            x2 = np.linspace(1, len(y2), len(y2))
            x = d1["date_short"][d1["group"] == g].reset_index(drop = True).iloc[168:516]
            ax.plot(x2, y2, label = wealth_groups[i] + "data", color = colors[i], linestyle = "--")
            ax.plot(x, y[0:len(y2),i], label = wealth_groups[i] + "model", color = colors[i])

        x = x.reset_index(drop=True)
        ax.set_xticks(x.iloc[0::20].index)
        ax.set_xticklabels(x.iloc[0::20], rotation = 90)
        ax.get_yaxis().set_visible(False)
        ax.set_ylim((-0.05,1))
        ax.legend(loc=(1.05, 0.45), frameon = False)
        ax.margins(0)
        ax.text(0,1.05, 'b', fontsize = 12)
