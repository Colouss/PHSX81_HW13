import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
np.random.seed(100) # The random seed used for the code
n = 1000 #generate the dataset with a small number of random noise
x = np.linspace(0, 2*np.pi, n)
y = np.sin(x) + np.random.normal(scale=0.1, size=n)
density = gaussian_kde(y, bw_method=0.1) # Estimate density using Gaussian Kernel density estimation, with the bw_method being the value of the density estimator parameter
grid = np.linspace(np.min(y), np.max(y), 1000) # Evaluate density at specific points on the function
estimated_density = density(grid)
plt.hist(y, density=True, alpha=0.5) # Plot estimated density and dataset
plt.plot(grid, estimated_density, color='red')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()
