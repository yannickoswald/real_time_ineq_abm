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


class Agent:
    def __init__(self, id, wealth, model):
        self.id = id
        scale_coeff = 150000
        self.wealth = float(powerlognorm.rvs(1.92, 2.08, size=1))*scale_coeff
        self.model = model
        ### variable that determines how much an agent is willing to trade/risk
        self.willingness_to_risk = random.uniform(0,0.1)
        self.num_links = 1

    def step(self, model):
        """Trade with linked agents"""
        # Get the neighbors of this agent
        neighbors = [n for n in model.graph.neighbors(self.id)]

        # Trade with agents that are reachable via a single intermediary
        for neighbor in neighbors:
                self.trade(other     = model.graph.nodes[neighbor]["agent"], 
                           concavity = model.concavity)
                
        self.wealth = self.wealth * (1 + model.economic_growth)
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
    def __init__(self, num_agents, initial_wealth, concavity, economic_growth):
        self.num_agents = num_agents
        self.agents = [Agent(i, initial_wealth, self) for i in range(num_agents)]
        self.graph = self.create_network()
        self.economic_growth = economic_growth
        self.wealth_data = list()
        self.concavity = concavity

    def create_network(self):
        """Create a graph with Barabasi-Albert degree distribution"""
        # Generate a network with m (desired number of edges each new node should have) = 2
        G = nx.barabasi_albert_graph(self.num_agents, 2)
        
        # Assign agents to nodes
        for i in range(self.num_agents):
            G.nodes[i]["agent"] = self.agents[i]
        return G
    
    
    def collect_wealth_data(self):
        """Collect wealth data over time as plot input"""
        time_step_data = list()
        for a in self.agents:
            time_step_data.append(a.wealth)
        self.wealth_data.append(time_step_data)
        
    
    def step(self):
        """Advance the model by one step"""
        # Randomly order agents and let them act in that order
        random_order = random.sample(self.agents, len(self.agents))
        for agent in random_order:
            agent.step(self)
            
        self.collect_wealth_data()   

    def plot_network(self):
        """Plot the network of agents"""
        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, with_labels=True,
                node_color='skyblue', node_size=1500,
                edge_cmap=plt.cm.Blues)
        plt.show()
        
    def plot_wealth_histogram(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,10))
        wealth_data = list()
        for a in self.agents:
            wealth_data.append(a.wealth)
        ax.hist(wealth_data)
        ax.set_xlabel("wealth")
        ax.set_ylabel("frequency")
        

    def plot_wealth_groups_over_time(self):
        labels = ["Top 1%", "Top 10%", "Middle 40%", "Bottom 50%"]
        groups_over_time = list()
        for i in range(0, len(self.wealth_data)):
            t = find_wealth_groups2(np.array(self.wealth_data[i]),
                                    sum(self.wealth_data[i]))[1]
            groups_over_time.append(t)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,10))
        y = np.vstack(groups_over_time)
        x = np.linspace(1, len(y), len(y))
        for i in range(0,4):
            ax.plot(x, y[:,i], label = labels[i])
        ax.set_xlabel("time")
        ax.set_ylabel("wealth share")
        ax.set_ylim((0,1))
        ax.legend()
        

                   
        
        
model = Model(500, initial_wealth=100, concavity=0.01, economic_growth = 0.02)  # 100 agents
for _ in range(360):  # Run for 10 steps
    model.step()
model.plot_network()
model.plot_wealth_histogram()
test = model.plot_wealth_groups_over_time()