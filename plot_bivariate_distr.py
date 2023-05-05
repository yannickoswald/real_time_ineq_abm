# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:16:47 2023

@author: earyo
"""

### https://python.quantecon.org/kalman.html
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
from scipy import linalg
import numpy as np
import matplotlib.cm as cm
from quantecon import Kalman, LinearStateSpace
from scipy.stats import norm
from scipy.integrate import quad
from numpy.random import multivariate_normal
from scipy.linalg import eigvals
import pandas as pd

with open('./data/test_bivariate_distr.csv') as f:
            data = pd.read_csv(f, encoding = 'unicode_escape', header = None)    
            
x = np.array(data.iloc[0,:])
y = np.array(data.iloc[3,:])

"""
# Set up the Gaussian prior density p
Σ = [[0.4, 0.3], [0.3, 0.45]]
Σ = np.matrix(Σ)
x_hat = np.matrix([0.2, -0.2]).T
# Define the matrices G and R from the equation y = G x + N(0, R)
G = [[1, 0], [0, 1]]
G = np.matrix(G)
R = 0.5 * Σ
# The matrices A and Q
A = [[1.2, 0], [0, -0.2]]
A = np.matrix(A)
Q = 0.3 * Σ
# The observed value of y
y = np.matrix([2.3, -1.9]).T

# Set up grid for plotting
x_grid = np.linspace(min(x), max(x), 100)
y_grid = np.linspace(min(y), max(y), 100)
X, Y = np.meshgrid(x_grid, y_grid)

"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_bivariate_normal(a, b):
    n = len(a)
    mean = [np.mean(a), np.mean(b)]
    cov = np.cov(a, b)
    x, y = np.meshgrid(np.linspace(np.min(a), np.max(a), n), np.linspace(np.min(b), np.max(b), n))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = multivariate_normal(mean=mean, cov=cov)
    fig, ax = plt.subplots()
    ax.contourf(x, y, rv.pdf(pos))
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_title('Bivariate Normal Distribution')
    plt.show()


plot_bivariate_normal(x, y)
