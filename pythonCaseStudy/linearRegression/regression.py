import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


'''generating and ploting normal numbers'''
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size = n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc = 0, scale = 1, size = n)

plt.figure()
plt.plot(x, y, 'o', ms = 5)

xx = np.array([0, 10])
plt.plot(xx, beta_0 + beta_1 * xx)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

def compute_rss(y_estimate, y):
  return sum(np.power(y-y_estimate, 2))

def estimate_y(x, b_0, b_1):
  return b_0 + b_1 * x
  
rss = compute_rss(estimate_y(x, beta_0, beta_1), y)