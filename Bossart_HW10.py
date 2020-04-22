## import modules
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
delta1 = 0.7          # known delta for this problem
n = 200
burn = 200

# part a
dist1 = np.random.normal(loc=mu1,
                         scale=sig,
                         size=int(n*delta1))
dist2 = np.random.normal(loc=mu2,
                         scale=sig,
                         size=int(n*(1-delta1)))
mix_vals = np.hstack((dist1, dist2))

# plotting for a
bns = np.linspace(4,11,14)
plt.figure()
sns.distplot(mix_vals,
             bins=bns,
             hist=True,
             kde=True,
             norm_hist= True,
             label = 'Sampling distribution') 
 
plt.xlabel('$y$ value')
plt.ylabel('Frequency')
plt.title('Sampling Distribution of Mixture Model $\delta = 0.7$')
plt.legend()

## part b
def mix_normal(vals, delta):
  return delta*stats.norm.pdf(vals, mu1, sig) + (1-delta)*stats.norm.pdf(vals, mu2, sig)

def likelihood(delta):
  return np.prod(mix_normal(mix_vals, delta))

m = 100000
deltas = np.zeros(m)          # this is where we will place our updated deltas
delta = delta_0 = 0.5         # some value between 0 and 1

for i in range(m):
    
    # grab from the prior/proposal
    delt_star = np.random.uniform(low=0, 
                                  high=1, 
                                  size=1)
    
    # compute Metropolis-Hastings ratio R
    R = likelihood(delt_star)/likelihood(delta)  # see ex 7.1 for justification
    
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(low=0, 
                                 high=1,
                                 size=1)  # determines if we accept or reject (think of this as prob)
    
    if rand_val <= np.min((R, 1)):
        delta = delt_star 
    else:
        delta = delta
        
    deltas[i] = delta


# plotting for b
deltas = deltas[burn:]         # burn in period
plt.figure()
sns.distplot(deltas,
             bins=np.linspace(0.5, 0.85, 30),
             kde=False,
             norm_hist=True,
             label = 'Posterior sample') 

# plt.hist(deltas, 
#          density=True, 
#          color='green',
#          bins=np.linspace(0.5, 0.85, 30),
#          rwidth=0.8,
#          label='Posterior sampling of $\delta$')
 
plt.xlabel('$\delta$ value')
plt.ylabel('Frequency')
plt.title('Sampling Distribution of $\delta$ using MCMC')
plt.legend()     


# path plot
xax = np.arange(200, m)
plt.figure()
plt.plot(xax, deltas, linewidth=1)
plt.title('Markov Chain path of $\delta$ with burn-in period')
plt.xlabel('Iteration')
plt.ylabel('Value')

                         

b_mean = np.mean(deltas)
b_var = np.var(deltas)
print('The sample mean of delta using Markov Chain Monte Carlo is ' + str(b_mean))
print('The sample variance of delta using Markov Chain Monte Carlo is ' + str(b_var))
print('using ' + str(m) + ' iterations')
print('and a burn-in period of ' + str(burn))




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
    
    
deltas = deltas[burn:]
plt.figure()
sns.distplot(deltas,
             bins=np.linspace(0.5, 0.85, 30),
             norm_hist=True,
             kde=False,
             label = 'Posterior sampling of $\delta$') 

plt.xlabel('$\delta$ value')
plt.ylabel('Frequency')
plt.title('Sampling Distribution of $\delta$ using Random Walk')
plt.legend()      

# path plot
plt.figure()
plt.plot(xax, deltas, linewidth=1)
plt.title('Random Walk path of $\delta$ with burn-in period')
plt.xlabel('Iteration')
plt.ylabel('Value')                           

c_mean = np.mean(deltas)
c_var = np.var(deltas)
print('The sample mean of delta using Markov Chain Monte Carlo with random walk is ' + str(c_mean))
print('The sample variance of delta using Markov Chain Monte Carlo with random walk is ' + str(c_var))
print('using ' + str(m) + ' iterations')
print('and a burn-in period of ' + str(burn))


# part d (logit reparameterization)
m=100000
deltas = np.zeros(m)          # this is where we will place our updated deltas
delta = delta_0 = logit(0.5)  # some value between 0 and 1

def mix2(vals, delta):
    return (expit(delta)*stats.norm.pdf(vals, loc=mu1, scale=sig) + (1-expit(delta))*stats.norm.pdf(vals, loc=mu2, scale=sig))
    
def jacob(delta):
    return ((1/(np.exp(delta)+1)-1/(np.exp(delta)+1)**2))

def likelihood2(delta):
    return np.prod(mix2(mix_vals, delta)) 

for i in range(m):
    
    # grab from the prior/proposal
    delt_star = delta + np.random.uniform(-1,1)
    
    # compute Metropolis-Hastings ratio R
    R = (likelihood2(delt_star)*np.abs(jacob(delt_star)))/(likelihood2(delta)*np.abs(jacob(delta)))
        
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(0,1)  # probability value
    
    if rand_val <= np.min((R,1)) and 0 <= expit(delt_star) <= 1:
        delta = delt_star
    else:
        delta = delta
        
    deltas[i] = expit(delta)

deltas = deltas[burn:]
plt.figure()
sns.distplot(deltas,
             bins=np.linspace(0.5, 0.85, 30),
             norm_hist=True,
             kde=False,
             label = 'Posterior sampling of $\delta$') 

plt.xlabel('$\delta$ value')
plt.ylabel('Frequency')
plt.title('Sampling Distribution of $\delta$ using Reparameterized Random Walk')
plt.legend()      

# path plot
plt.figure()
plt.plot(xax, deltas, linewidth=1)
plt.title('Random Walk Reparameterization path of $\delta$ with burn-in period')
plt.xlabel('Iteration')
plt.ylabel('Value')                           

d_mean = np.mean(deltas)
d_var = np.var(deltas)
print('The sample mean of delta using Markov Chain Monte Carlo with reparameterized random walk is ' + str(d_mean))
print('The sample variance of delta using Markov Chain Monte Carlo with reparameterized random walk is ' + str(d_var))
print('using ' + str(m) + ' iterations')
print('and a burn-in period of ' + str(burn))  

    


# number 2

def f(x):
  return delta*stats.norm.pdf(x, mu1, sig) + (1-delta)*stats.norm.pdf(x, mu2, sig)


# part a
m=10000
xs1 = np.zeros(m)           
x = x_0 = 0 
delta = 0.7

## first starting value
for i in range(m):
    
    # grab from the proposal
    x_star = stats.norm.rvs(loc=x, scale=0.01, size=1)
    
    # compute Metropolis-Hastings ratio R
    R = (f(x_star)*stats.norm.pdf(x, loc=x, scale=0.01))/(f(x)*stats.norm.pdf(x_star, loc=x, scale=0.01))
        
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(0,1)
    
    if rand_val <= np.min((R,1)):
        x = x_star
    else:
        x = x_star
        
    xs1[i] = x
    
## second starting value
xs2 = np.zeros(m)           
x = x_0 = 7 
for i in range(m):
    
    # grab from the proposal
    x_star = stats.norm.rvs(loc=x, scale=0.01, size=1)
    
    # compute Metropolis-Hastings ratio R
    R = (f(x_star)*stats.norm.pdf(x, loc=x, scale=0.01))/(f(x)*stats.norm.pdf(x_star, loc=x, scale=0.01))
        
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(0,1)
    
    if rand_val <= np.min((R,1)):
        x = x_star
    else:
        x = x_star
        
    xs2[i] = x
    
## third starting value
xs3 = np.zeros(m)           
x = x_0 = 15 
for i in range(m):
    
    # grab from the proposal
    x_star = stats.norm.rvs(loc=x, scale=0.01, size=1)
    
    # compute Metropolis-Hastings ratio R
    R = (f(x_star)*stats.norm.pdf(x, loc=x, scale=0.01))/(f(x)*stats.norm.pdf(x_star, loc=x, scale=0.01))
        
    # sample a value for X(t+1) 
    rand_val = np.random.uniform(0,1)
    
    if rand_val <= np.min((R,1)):
        x = x_star
    else:
        x = x_star
        
    xs3[i] = x
    
    
# plotting the chains
fig1 = plt.figure()
plt.title('Markov Chain with $x^{(0)} = 0$')
plt.plot(np.linspace(0, m, len(xs1)), xs1)
plt.xlabel('Iteration')
plt.ylabel('$x$ value')

fig2 = plt.figure()
plt.title('Markov Chain with $x^{(0)} = 7$')
plt.plot(np.linspace(0, m, len(xs2)), xs2)
plt.xlabel('Iteration')
plt.ylabel('$x$ value')

fig3 = plt.figure()
plt.title('Markov Chain with $x^{(0)} = 15$')
plt.plot(np.linspace(0, m, len(xs3)), xs3)
plt.xlabel('Iteration')
plt.ylabel('$x$ value')

# plotting the histograms
