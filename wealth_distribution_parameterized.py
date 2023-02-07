# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:59:25 2023

@author: earyo
"""

#%%

### This script will fit a power log-normal distribution to wealth data in the USA 

# Distribution taken from
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlognorm.html#scipy.stats.powerlognorm


import numpy as np
from scipy.stats import powerlognorm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

loc = 100000
scale = 100000

c, s = 1.1, 0.446
mean, var, skew, kurt = powerlognorm.stats(c, s, loc, scale, moments='mvsk')


x = np.linspace(powerlognorm.ppf(0.01, c, s, loc, scale),
                powerlognorm.ppf(0.99, c, s, loc, scale), 100)
ax.plot(x, powerlognorm.pdf(x, c, s, loc, scale),
       'r-', lw=5, alpha=0.6, label='powerlognorm pdf')


rv = powerlognorm(c, s, loc, scale)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')


vals = powerlognorm.ppf([0.001, 0.5, 0.999], c, s, loc, scale)
np.allclose([0.001, 0.5, 0.999], powerlognorm.cdf(vals, c, s, loc, scale))

r = powerlognorm.rvs(c, s, loc, scale, size=1000)


ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])
ax.legend(loc='best', frameon=False)
plt.show()