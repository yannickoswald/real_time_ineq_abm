# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:44:01 2023

@author: earyo
"""
import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 1, 10)
Y = np.linspace(0, 1, 10)

x,y = np.meshgrid(X,Y)

f1 = np.cos(x*y)
f2 = x-y

plt.contour(x,y,f2,colors='red')
plt.contour(x,y,f1,colors='blue')
plt.show()