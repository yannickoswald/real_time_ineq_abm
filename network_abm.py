# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:41:18 2023

@author: earyo
"""


import networkx as nx
import matplotlib.pyplot as plt
import random as random
from inequality_metrics import find_wealth_groups2
import numpy as np
from scipy.stats import powerlognorm
import pandas as pd

class Agent:
    def __init__(self, id, model):
        self.id = id
        # TODO: Maybe this could be provided to the agent when it is
        # instantiated in the model?
        # Or even used as as model parameter?
        scale_coeff = 150000
        self.wealth = float(powerlognorm.rvs(1.92, 2.08, size=1))*scale_coeff
        self.model = model
        ### variable that determines how much an agent is willing to trade/risk
        self.willingness_to_risk = random.uniform(0, 0.1)
        self.num_links = 1

    def step(self, model):
        """Trade with linked agents"""
        # Get the neighbors of this agent
        neighbors = [n for n in model.graph.neighbors(self.id)]

        # Trade with agents that are reachable via a single intermediary
        for neighbor in neighbors:
                self.trade(other     = model.graph.nodes[neighbor]["agent"], 
                           concavity = model.concavity)
                
        self.wealth = self.wealth * (1 + model.growth_rate)
        self.num_links = self.count_links(model = self.model)

    def trade(self, other, concavity):
        """Perform a trade between this agent and another agent"""
        # The fraction of wealth to be traded is limited to the wealth of the poorer agent
        a = self.willingness_to_risk
        b = other.willingness_to_risk
        fraction = random.uniform(0, min(a*self.wealth, b*other.wealth))

        # The probability of winning is proportional to the wealth of the agent
        self_win_probability = ((self.num_links / (self.num_links + other.num_links))**concavity) + np.random.normal(0,0.2)
        if random.random() < self_win_probability:
            # Self wins the trade
            self.wealth += fraction
            other.wealth -= fraction
        else:
            # Other wins the trade
            self.wealth -= fraction
            other.wealth += fraction
            
    def count_links(self, model):
        """Count the number of links this agent has"""
        return len(list(model.graph.neighbors(self.id)))
        

class Model:
    def __init__(self, num_agents, concavity, growth_rate, start_year):
        self.num_agents = num_agents
        self.agents = [Agent(i, self) for i in range(num_agents)]
        self.graph = self.create_network()
        self.growth_rate = growth_rate
        self.wealth_data = list()
        self.concavity = concavity
        self.start_year = start_year ## doesn't do anything currently
        self.time = 0
        self.macro_state = None
        self.micro_state = None

    def create_network(self):
        """Create a graph with Barabasi-Albert degree distribution"""
        # Generate a network with m (desired number of edges each new node should have) = 2
        G = nx.barabasi_albert_graph(self.num_agents, 2)
        
        # Assign agents to nodes
        for i in range(self.num_agents):
            G.nodes[i]["agent"] = self.agents[i]
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
        # TODO: Do we need subplots here?
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
        
        print(d1)
            
        colors = ["tab:red", "tab:blue", "grey", "y"]
        labels = ["Top 1%", "Top 10%", "Middle 40%", "Bottom 50%"]
        groups_over_time = list()
        for i in range(0, len(self.wealth_data)):
            a = np.array(self.wealth_data[i])
            b = sum(self.wealth_data[i])
            t = find_wealth_groups2(a, b)[1]
            groups_over_time.append(t)
        y = np.vstack(groups_over_time)
        x = np.linspace(1, len(y), len(y))
        
        for i in range(0,4):
            key = d1["group"] == labels[i]
            y2 = d1["real_wealth_share"][key].reset_index(drop = True).iloc[168:516]
            x2 = np.linspace(1, len(y2), len(y2))
   
            ax.plot(x2, y2, label = labels[i] + "data", color = colors[i], linestyle = "--")
            ax.plot(x, y[:,i], label = labels[i] + "model", color = colors[i])
        ax.set_xlabel("time")
        #ax.set_ylabel("wealth share")
        ax.set_ylim((0,1))
        ax.legend(loc=(1.05, 0.5))

