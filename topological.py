import numpy as np
from imageio.v3 import imread
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import reg_class

n_deg_max = 30 # max polynomial degree
lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0] # lambdas to try with Ridge regression
downsample_scale = 50

# Load the terrain and get axis
terrain_raw = np.asarray(imread('SRTM_data_Norway_1.tif'))
x_raw = np.arange(terrain_raw.shape[1])
y_raw = np.arange(terrain_raw.shape[0])

# Downsample data and axes
terrain = terrain_raw[0::downsample_scale, 0::downsample_scale]
x = x_raw[0::downsample_scale]
y = y_raw[0::downsample_scale]
x_, y_ = np.meshgrid(x, y)

# # Plot terrain
plt.figure()
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plots/terrain.pdf')

# Plot terrain as 3d surface
fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
ax.plot_surface(x_, y_, terrain, cmap = cm.coolwarm)
ax.view_init(70, -45, 0)
ax.axis('equal')
fig.savefig("plots/terrain_3d.pdf")

xy = np.stack((np.ravel(x_),np.ravel(y_)), axis = -1) # formatting needed to set up the design matrix

model = reg_class.regression_class(xy, terrain.flatten(), n_deg_max, lmbda)

# Do regression
model.ols_regression()
model.ridge_regression()
model.lasso_regression()

# Plot regression results
model.plot_ols_results()
model.plot_ridge_results()
model.plot_lasso_results()

# # Make predictions and plot
pol_degree = 23
prediction = np.reshape(model.predict_ols(pol_degree), (len(y), len(x))) # ols prediction
fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
ax.plot_surface(x_, y_, prediction, cmap = cm.coolwarm)
ax.view_init(70, -45, 0)
ax.axis('equal')
fig.savefig(f"plots/topological_pred_{pol_degree}.pdf")