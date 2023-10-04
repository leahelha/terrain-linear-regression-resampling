import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mean_squared_error
import reg_class

def FrankeFunction(x,y):
    '''Calculates the two-dimensional Franke's function.'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def add_noise(data, std):
    '''Adds noise from a normal distribution N(0,std^2) to an array of any shape.'''
    noise_matrix = np.random.normal(0, std, data.shape)
    return data + noise_matrix



### Set up dataset ###
n = 101 # number of points along one axis, total number of points will be n^2
rng = np.random.default_rng(seed = 25) # seed to ensure same numbers over multiple runs
x = np.sort(rng.random((n, 1)), axis = 0)
y = np.sort(rng.random((n, 1)), axis = 0)
x_, y_ = np.meshgrid(x, y)
xy = np.stack((np.ravel(x_),np.ravel(y_)), axis = -1) # formatting needed to set up the design matrix
z = add_noise(FrankeFunction(x_, y_), 0.1)
# z = FrankeFunction(x_, y_)

# Plot Franke function
fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
ax.plot_surface(x_, y_, z, cmap = cm.coolwarm)
fig.savefig("plots/franke.pdf")



### Choose parametres for regression ###
n_deg_max = 5 # max polynomial degree
lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0] # lambdas to try with Ridge regression



### Do regression with our own code ###
model = reg_class.regression_class(xy, z.flatten(), n_deg_max, lmbda)
model.ols_regression()
model.ridge_regression()
model.lasso_regression()



### Plot prediction
pol_degree = 5
prediction = np.reshape(model.predict_ols(pol_degree), (len(x), len(y)))
fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
ax.plot_surface(x_, y_, prediction, cmap = cm.coolwarm)
fig.savefig("plots/franke_pred.pdf")



### Do regression with scikit-learn ###
# Lists to store results
beta_ols = [0]*n_deg_max
beta_ridge = [0]*n_deg_max
mse_ols = [0]*n_deg_max
mse_ridge = [0]*n_deg_max

# Get data from model
X_train_scaled = model.X_train_scaled
y_train_scaled = (model.y_train_scaled).reshape(-1,1)

for pol_degree in range(1,n_deg_max+1): # for each polynomial degree
    # Pick out relevant part of design matrix
    N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
    X_train_scaled_N = X_train_scaled[:, 0:N]

    # For OLS, find beta-values and MSE using scikit-learn
    m_ols = LinearRegression().fit(X_train_scaled_N, y_train_scaled)
    beta_ols[pol_degree-1] = m_ols.coef_[0]
    mse_ols[pol_degree-1] = mean_squared_error(m_ols.predict(X_train_scaled_N), y_train_scaled)

    beta_lmbda = []
    mse_lmbda = []

    for i in range(len(lmbda)): # for each lambda value
        # For Ridge, find beta-values and MSE using scikit-learn
        m_ridge = Ridge(alpha = lmbda[i]).fit(X_train_scaled_N, y_train_scaled)
        beta_lmbda.append(m_ridge.coef_[0])
        mse_lmbda.append(mean_squared_error(m_ridge.predict(X_train_scaled_N), y_train_scaled))

    beta_ridge[pol_degree-1] = beta_lmbda
    mse_ridge[pol_degree-1] = mse_lmbda



### Compare results from our own code and scikit-learn ###
tol = 1e-7 # tolerance limit

# Test for OLS
beta_test = True
mse_test = True

# Loop through each beta and MSE value, and check wether the difference
# between our values and scikit-learn's values is less that tol
for i in range(len(beta_ols)):
    if (np.abs(mse_ols[i] - model.ols["mse_train"][i]) > tol):
        mse_test = False
    for j in range(len(beta_ols[i])):
        if (np.abs(beta_ols[i][j] - model.ols["beta"][i][j]) > tol):
            beta_test = False

# Print results
print("\n--- TEST OLS ---")
print(f"MSE_train: {mse_test}")
print(f"Beta: {beta_test}")

# Test for Ridge
beta_test = True
mse_test = True

# Loop through each beta and MSE value, and check wether the difference
# between our values and scikit-learn's values is less that tol
for i in range(len(beta_ridge)):
    for j in range(len(beta_ridge[i])):
        if (np.abs(mse_ridge[i][j] - model.ridge["mse_train"][i][j]) > tol):
            mse_test = False
        for k in range(len(beta_ridge[i][j])):
            if (np.abs(beta_ridge[i][j][k] - model.ridge["beta"][i][j][k]) > tol):
                beta_test = False

# Print results
print("\n--- TEST RIDGE ---")
print(f"MSE_train: {mse_test}")
print(f"Beta: {beta_test}\n")