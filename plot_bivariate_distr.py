
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:16:47 2023
@author: earyo
"""

### https://python.quantecon.org/kalman.html
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
import matplotlib.cm as cm
#from matplotlib.ticker import LogFormatterSciNotation
"""
with open('./data/test_bivariate_distr.csv') as f:
            data = pd.read_csv(f, encoding = 'unicode_escape', header = None)    
            
x = np.array(data.iloc[0,:])
y = np.array(data.iloc[3,:])
x_mean = np.mean(x)
y_mean = np.mean(y)
x_hat = np.matrix([x_mean, y_mean]).T
C = np.cov(x,y)
# Set up grid for plotting
x_grid = np.linspace(0, max(x), 100)
y_grid = np.linspace(min(y)/2, max(y), 100)
X, Y = np.meshgrid(x_grid, y_grid)
"""
def bivariate_normal(x, y, σ_x=1.0, σ_y=1.0, μ_x=0.0, μ_y=0.0, σ_xy=0.0):
    """
    Compute and return the probability density function of bivariate normal
    distribution of normal random variables x and y
    Parameters
    ----------
    x : array_like(float)
        Random variable
    y : array_like(float)
        Random variable
    σ_x : array_like(float)
          Standard deviation of random variable x
    σ_y : array_like(float)
          Standard deviation of random variable y
    μ_x : scalar(float)
          Mean value of random variable x
    μ_y : scalar(float)
          Mean value of random variable y
    σ_xy : array_like(float)
           Covariance of random variables x and y
    """

    x_μ = x - μ_x
    y_μ = y - μ_y
    ρ = σ_xy / (σ_x * σ_y)
    z = x_μ**2 / σ_x**2 + y_μ**2 / σ_y**2 - 2 * ρ * x_μ * y_μ / (σ_x * σ_y)
    denom = 2 * np.pi * σ_x * σ_y * np.sqrt(1 - ρ**2)
    return np.exp(-z / (2 * (1 - ρ**2))) / denom

def gen_gaussian_plot_vals(μ, C, X, Y):
    "Z values for plotting the bivariate Gaussian N(μ, C)"
    X = X
    Y = Y
    m_x, m_y = float(μ[0]), float(μ[1])
    s_x, s_y = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
    s_xy = C[0, 1]
    return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)

# Plot the figure
def plot_bivariate_normal(x_hat, C, X, Y, varname1:str , varname2: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid()
    X = X
    Y = Y
    Z = gen_gaussian_plot_vals(x_hat, C, X, Y)
    ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.coolwarm)
    cs = ax.contour(X, Y, Z, 6, colors="black")
    #Contour levels are a probability density
    ax.clabel(cs,levels = cs.levels, inline=1, fontsize=10)
    ax.set_xlabel(varname1, fontsize = 14)
    ax.set_ylabel(varname2, fontsize = 14)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #ax.set_xticklabels([0, 0.5, 1, 1.5,2,2.5,3,3.5])
    #ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    #ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    plt.show()

#plot_bivariate_normal(x_hat, C, X, Y, "$ Wealth per adult Top 1%", "$ Weatlh per adult Bottom 50%")



#https://stackoverflow.com/questions/10490302/how-do-you-create-a-legend-for-a-contour-plot-in-matplotlib

