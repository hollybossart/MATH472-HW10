## import stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


## problem 7.1 from the textbook
np.random.seed(0)

mu1 = 7
mu2 = 10
sig = 0.5
delta1 = 0.7
n = 200

# part a
dist1 = np.random.normal(loc=mu1, scale=sig, size=int(n*delta1))
dist2 = np.random.normal(loc=mu2, scale=sig, size=int(n*(1-delta1)))
mix_vals = np.hstack((dist1, dist2))
plt.hist(mix_vals)  ## TODO: make prettier
plt.xlabel('Value')
plt.ylabel('Frequency')

# part b
def mix_normal(vals, delta):
  return delta*stats.norm.pdf(vals, mu1, sig) + (1-delta)*stats.norm.pdf(vals, mu2, sig)

def likelihood(delta):
  return np.prod(mix_normal(mix_vals, delta))

m = 200000
deltas = np.zeros(m)          # this is where we will place our updated deltas
delta = delta_0 = 0.5         # some value between 0 and 1

for i in range(m):
    
    # grab from the prior/proposal
    delt_star = np.random.uniform(low=0, high=1, size=1)
    
    # compute Metropolis-Hastings ratio R
    R = likelihood(delt_star)/likelihood(delta)  # see ex 7.1 for justification
    
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(low=0, high=1, size=1)
    
    if rand_val < R:
        delta = delt_star
        
    deltas[i] = delta

plt.figure()
plt.hist(deltas)

# part c (random walk chain)
deltas = np.zeros(m)          # this is where we will place our updated deltas
delta = delta_0 = 0.5         # some value between 0 and 1

for i in range(m):
    
    # grab from the prior/proposal
    delt_star = delta + stats.uniform.rvs(loc=-1, scale=2, size =1)
    
    # compute Metropolis-Hastings ratio R
    R = likelihood(delt_star)/likelihood(delta)  # see ex 7.1 for justification
    
    if R < 0 or R > 1:
        R = 0
        
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(low=0, high=1, size=1)
    
    if rand_val < R:
        delta = delt_star
        
    deltas[i] = delta

plt.figure()
plt.hist(deltas)

    

    
    