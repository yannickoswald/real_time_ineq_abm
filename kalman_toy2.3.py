# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:02:49 2022

@author: earyo
"""

### this is an exercise script with much copied from
### https://python.quantecon.org/kalman.html

### load all necessary libraries 
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

#%% 

# the following function is copied from
#https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour/18309914#18309914

def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours

#%%

### read data 

data = pd.read_csv (r'C:\Users\earyo\Dropbox\Arbeit\postdoc_leeds\Data assimilation toy examples python\china_data.csv',  encoding='latin-1')

### amend data to include 10 observations of x_1 and x_2 per time step
### amend dataframe
#for j in range(2):
 #   for i in range(10):
  #        data["obs_x" + str(j+1) + "_" + str(i+1)] = np.nan
          

### draw from normal distr. with specified parameters

#observations_1 = np.zeros((17,10))
#observations_2 = np.zeros((17,10))
#for j in range(len(data)):
#    for i in range(10):
 #       observations_1[j,i] = np.random.normal(data["GDPpc"][j], data["gdp_std"][j])
  #      observations_2[j,i] = np.random.normal(data["life_satisfaction"][j], data["life_satisfaction_std"][j])
###

### compute covariance matrices for all 17 years in the data 2004 - 2020
observations_1 = np.array(data.iloc[:,11:21])    
observations_2 = np.array(data.iloc[:,21:31])

covariance_matrices = np.zeros((2*17, 2*17))
for i in range(1,17+1):
    X = np.stack((observations_1[i-1,:], observations_2[i-1,:]), axis=0)
    covariance_matrices[i*2-2:i*2,i*2-2:i*2]= np.cov(X)
    
### calculate means from (fake) data observations 

fake_mean_gdp = np.mean(observations_1, axis = 1)
fake_mean_ls = np.mean(observations_2, axis = 1)


#%% 

###### CONDUCT FILTERING STEP FOR 2004 ########


# Set up the Gaussian prior density p
Σ =  covariance_matrices[0:2,0:2]
Σ = np.matrix(Σ)
x_hat = np.matrix([fake_mean_gdp[0], fake_mean_ls[0]]).T



# Define the matrices G and R from the equation y = G x + N(0, R)
G = [[1, 0], [0, 1]] ## based on linear regression between GDPpc and life satisfaction
G = np.matrix(G)
R = 0.5 * Σ

# The observed value 
y = np.matrix([4817.21, 4.5605]).T


# Set up grid for plotting
x_grid = np.linspace(4700, 4900, 1001)
y_grid = np.linspace(3.5, 6, 1000)
X, Y = np.meshgrid(x_grid, y_grid)


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


def gen_gaussian_plot_vals(μ, C):
    "Z values for plotting the bivariate Gaussian N(μ, C)"
    m_x, m_y = float(μ[0]), float(μ[1])
    s_x, s_y = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
    s_xy = C[0, 1]
    return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)



fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

Z = gen_gaussian_plot_vals(x_hat, Σ)
ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.jet)
cs = ax.contour(X, Y, Z, 6, colors="black")
#ax.clabel(cs, inline=1, fontsize=8)
ax.set_xlabel("GDP per capita", fontsize=16)
ax.set_ylabel("Life satisfaction (reported)", fontsize=16)
ax.set_title("Chinese economy 2004 (state variables)", fontsize=20)
ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")
ax.annotate('new observation', xy=(float(y[0] +8), float(y[1])-.03), xytext=(4850, 4.25),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize = 16)

ax.annotate(' bivariate \n normal distribution', xy=(4775, 5), xytext=(4701, 5.5),
 arrowprops=dict(facecolor='black', shrink=0.05), fontsize = 16)

plt.show()
plt.close()

assert x_hat[0] == 4.78905e+03
assert x_hat[1] == 4.76946e+00

assert round(Σ[0,0],2) == 1149.86
assert round(Σ[0,1],2) == round(18.7159,2)
assert round(Σ[1,0],2) == round(18.7159,2)
assert round(Σ[1,1],2) == round(0.336931,2)


#%%
### ACTUAL FILTERING STEP AND PLOT

fig2, ax = plt.subplots(figsize=(10, 8))
ax.grid()

Z = gen_gaussian_plot_vals(x_hat, Σ)
cs1 = ax.contour(X, Y, Z, 6, colors="black")
#ax.clabel(cs1, inline=1, fontsize=10)


M = Σ * G.T * linalg.inv(G * Σ * G.T + R)



x_hat_F = x_hat + M * (y - G * x_hat)


Σ_F = Σ - M * G * Σ
new_Z = gen_gaussian_plot_vals(x_hat_F, Σ_F)
cs2 = ax.contour(X, Y, new_Z, 6, colors="black")
#ax.clabel(cs2, inline=1, fontsize=10)
ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap=cm.jet)
ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")
ax.set_xlabel("GDP per capita", fontsize=16)
ax.set_ylabel("Life satisfaction (reported)", fontsize=16)
ax.set_title("Chinese economy 2004 (state variables)", fontsize=20)

plt.show()
plt.close()

assert x_hat[0] == 4.78905e+03
assert x_hat[1] == 4.76946e+00
assert round(Σ[0,0],2) == 1149.86
assert round(Σ[0,1],2) == round(18.7159,2)
assert round(Σ[1,0],2) == round(18.7159,2)
assert round(Σ[1,1],2) == round(0.336931,2)

listcontours1 = get_contour_verts(cs1)
listcontours1.pop(0)
listcontours1.pop(-1)

assert abs(np.mean(listcontours1[4][0][:,1])/x_hat[1]-1) < 0.01



#%% PREDICTION STEP 
## 
#implement in our case the model is the determined growth rates of GDPpc and 
### life satisfaction 
# The matrices A and Q
A = [[1.107, 0], [0, 1.066]]
A = np.matrix(A)
###uncertainty in economic forecast
## as addition because only element 11 is supposed to be altered
UEF =  np.matrix([[2000, 0], [0, 0]])
Q = UEF + Σ

assert x_hat[0] == 4.78905e+03
assert x_hat[1] == 4.76946e+00




#%%

##### DUMMY plot for density 1

fig0, ax0 = plt.subplots(figsize=(10, 8))
# Set up grid for plotting
x_grid_0 = np.linspace(4700, 4900, 1001)
y_grid_0 = np.linspace(3.5, 6, 1000)

X2, Y2 = np.meshgrid(x_grid_0, y_grid_0)
#ax0.grid()
Z = gen_gaussian_plot_vals(x_hat, Σ)
cs3 = ax.contour(X2, Y2, Z, 5, colors="black")
plt.close()


#ax.clabel(cs1, inline=1, fontsize=10)
listcontours1 = get_contour_verts(cs3)
listcontours1.pop(0)
listcontours1.pop(-1)

assert abs(np.mean(listcontours1[4][0][:,1])/x_hat[1]-1) < 0.01

#%%

##### DUMMY plot for density 2
fig00, ax00 = plt.subplots(figsize=(10, 8))
# Set up grid for plotting
x_grid_00 = np.linspace(4700, 4900, 1001)
y_grid_00 =  np.linspace(3.5, 6, 1000)
X_00, Y_00 = np.meshgrid(x_grid_00, y_grid_00)
ax00.grid()

M = Σ * G.T * linalg.inv(G * Σ * G.T + R)
x_hat_F = x_hat + M * (y - G * x_hat)
Σ_F = Σ - M * G * Σ
Z_F = gen_gaussian_plot_vals(x_hat_F, Σ_F)

cs4 = ax00.contour(X_00, Y_00, Z_F, 5, colors="black")
plt.close()
#ax.clabel(cs1, inline=1, fontsize=10)
listcontours2 = get_contour_verts(cs4)
listcontours2.pop(0)
listcontours2.pop(-1)

assert abs(np.mean(listcontours2[4][0][:,1])/x_hat_F[1]-1) < 0.01

#%%


##### DUMMY plot for density 3
fig000, ax000 = plt.subplots(figsize=(10, 8))

# Set up grid for plotting
x_grid_000 = np.linspace(4500, 5500, 1001)
y_grid_000 = np.linspace(3, 8, 1000)
X, Y = np.meshgrid(x_grid_000, y_grid_000)
ax000.grid()

new_x_hat = A * x_hat_F
new_Σ = A * Σ_F * A.T + Q
new_Z = gen_gaussian_plot_vals(new_x_hat, new_Σ)
cs5 = ax000.contour(X, Y, new_Z, 5, colors="black")
plt.show()
plt.close()
#ax.clabel(cs1, inline=1, fontsize=10)
listcontours3 = get_contour_verts(cs5)
listcontours3.pop(0)
listcontours3.pop(-1)


assert abs(np.mean(listcontours3[3][0][:,1])/new_x_hat[1]-1) < 0.01

#%%


# Density 1
fig3, ax1 = plt.subplots(figsize=(10, 8))
# Set up grid for plotting
x_grid = np.linspace(4500, 5500, 1001)
y_grid = np.linspace(3, 7, 1000)
X, Y = np.meshgrid(x_grid, y_grid)
ax1.grid()

Z = gen_gaussian_plot_vals(x_hat, Σ)
for i in range(len(listcontours1)): ax1.plot(listcontours1[i][0][:,0], listcontours1[i][0][:,1], color = "black")


# Density 2
#fig, ax2 = plt.subplots(figsize=(10, 8))
# Set up grid for plotting
#x_grid = np.linspace(4000, 7000, 1001)
#y_grid = np.linspace(0, 10, 1000)
#X, Y = np.meshgrid(x_grid, y_grid)
#ax1.grid()

M = Σ * G.T * linalg.inv(G * Σ * G.T + R)
x_hat_F = x_hat + M * (y - G * x_hat)
Σ_F = Σ - M * G * Σ
Z_F = gen_gaussian_plot_vals(x_hat_F, Σ_F)
for i in range(len(listcontours2)): ax1.plot(listcontours2[i][0][:,0], listcontours2[i][0][:,1], color = "black")

# Density 3
new_x_hat = A * x_hat_F
new_Σ = A * Σ_F * A.T + Q
new_Z = gen_gaussian_plot_vals(new_x_hat, new_Σ)
cs3 = ax1.contour(X, Y, new_Z, 5, colors="black")
#ax.clabel(cs3, inline=1, fontsize=10)
#for i in range(len(listcontours3)): ax1.plot(listcontours3[i][0][:,0], listcontours3[i][0][:,1], color = "black")


ax1.contourf(X, Y, new_Z, 5, alpha=0.6, cmap=cm.jet)
#ax1.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")
ax1.set_xlabel("GDP per capita", fontsize=16)
ax1.set_ylabel("Life satisfaction (reported)", fontsize=16)
ax1.set_title("Chinese economy 2005 (state variables)", fontsize=20)

plt.show()


#%%


import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-5.0, 3.0, 100)
ylist = np.linspace(-5.0, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()