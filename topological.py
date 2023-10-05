import numpy as np
from imageio.v3 import imread
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import reg_class

n_deg_max = 10 # max polynomial degree
lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0] # lambdas to try with Ridge regression

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')

# Plot terrain
plt.figure()
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plots/terrain.pdf')

# Plot terrain as 3d surface
x = np.arange(terrain1.shape[1])
y = np.arange(terrain1.shape[0])
x_, y_ = np.meshgrid(x, y)
xy = np.stack((np.ravel(x_),np.ravel(y_)), axis = -1) # formatting needed to set up the design matrix
fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
ax.plot_surface(x_, y_, terrain1, cmap = "gray")
ax.view_init(70, -45, 0)
ax.axis('equal')
fig.savefig("plots/terrain_3d.pdf")

model = reg_class.regression_class(xy, terrain1.flatten(), n_deg_max, lmbda)

# Do regression
model.ols_regression()
# model.ridge_regression()
# model.lasso_regression()

# Plot results
model.plot_ols_results()
# model.plot_ridge_results()
# model.plot_lasso_results()