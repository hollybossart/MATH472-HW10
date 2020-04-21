## import stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import logit, expit


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

m = 10000
deltas = np.zeros(m)          # this is where we will place our updated deltas
delta = delta_0 = 0.5         # some value between 0 and 1

for i in range(m):
    
    # grab from the prior/proposal
    delt_star = np.random.uniform(low=0, high=1, size=1)
    
    # compute Metropolis-Hastings ratio R
    R = likelihood(delt_star)/likelihood(delta)  # see ex 7.1 for justification
    
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(low=0, high=1, size=1) # determines if we accept or reject (think of this as prob)
    
    if rand_val <= np.min((R, 1)):
        delta = delt_star
        
    else:
        delta = delta
        
    deltas[i] = delta

plt.figure()
sns.distplot(deltas, hist=True, kde=True)


# part c (random walk chain)
m=100000
deltas = np.zeros(m)          # this is where we will place our updated deltas
delta = delta_0 = 0.5         # some value between 0 and 1

for i in range(m):
    
    # grab from the prior/proposal
    delt_star = delta + np.random.uniform(-1,1)
    
    # compute Metropolis-Hastings ratio R
    R = likelihood(delt_star)/likelihood(delta)
        
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(0,1)
    
    if rand_val <= np.min((R,1)) and delt_star <= 1 and delt_star >= 0:
        delta = delt_star
    else:
        delta = delta
        
    deltas[i] = delta

plt.figure()
plt.hist(deltas)


# part d (logit reparameterization)
m=100000
deltas = np.zeros(m)          # this is where we will place our updated deltas
delta = delta_0 = logit(0.5)  # some value between 0 and 1

def jacob(delta):
    return np.abs(1/(np.exp(delta)+1)-1/(np.exp(delta)+1)**2)

for i in range(m):
    
    # grab from the prior/proposal
    delt_star = delta + np.random.uniform(-1,1)
    
    # compute Metropolis-Hastings ratio R
    R = likelihood(delt_star)/likelihood(delta)
        
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(0,1)
    
    if rand_val <= np.min((R,1)) and delt_star <= 1 and delt_star >= 0:
        delta = delt_star
    else:
        delta = delta
        
    deltas[i] = delta

plt.figure()
plt.hist(deltas)   

    
    