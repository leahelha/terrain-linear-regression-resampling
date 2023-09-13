import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm

def FrankeFunction(x,y):
    '''Calculates the two-dimensional Franke's function.'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def add_noise(data):
    '''Adds noise from a normal distribution N(0,1) to an array of any shape.'''
    noise_matrix = np.random.normal(0, 1, data.shape)
    return data + noise_matrix

def beta_ols(X, y):
    '''Given the design matrix X and the output y, calculates the coefficients beta using OLS.'''
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def beta_ridge(X, y, lmbda):
    '''Given the design matrix X, the output y and the parameter lmbda, calculates the coefficients beta using OLS.'''
    n = np.shape(X)[1]
    beta = np.linalg.inv(X.T @ X + lmbda*np.eye(n)) @ X.T @ y
    return beta

def mse_own(y_tilde, y):
    '''Calculates the mean square error of a prediction y_tilde.'''
    mse = 1/len(y) * np.sum((y-y_tilde)**2)
    return mse

def r2_own(y_tilde, y):
    '''Calculates the R^2 score of a prediction y_tilde.'''
    a = np.sum((y-y_tilde)**2)
    b = np.sum((y-np.mean(y))**2)
    return 1 - a/b

def fit_predict_ols(x, y, pol_degree):
    '''For a given polynomial order, makes and trains an OLS model and calculates MSE for both training and test data.'''
    
    # Make design matrix and split into training and test data
    X = PolynomialFeatures(pol_degree, include_bias = False).fit_transform(x) # without intercept
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3) # random_state gives same partition across multiple function calls
    
    # Scale data by subtracting mean
    X_scaler = np.mean(X_train, axis = 0)
    y_scaler = np.mean(y_train)
    X_train_scaled = X_train - X_scaler
    y_train_scaled = y_train - y_scaler
    X_test_scaled = X_test - X_scaler

    # Fit parametres
    beta = beta_ols(X_train_scaled, y_train_scaled)

    # Make predictions
    y_train_pred = X_train_scaled @ beta + y_scaler
    y_test_pred = X_test_scaled @ beta + y_scaler

    # Calculate MSE and R^2 for both training and test data
    mse_train = mse_own(y_train_pred, y_train)
    mse_test = mse_own(y_test_pred, y_test)
    r2_train = r2_own(y_train_pred, y_train)
    r2_test = r2_own(y_test_pred, y_test)

    return beta, mse_train, mse_test, r2_train, r2_test

def fit_predict_ridge(x, y, pol_degree, lmbda):
    '''For a given polynomial order, makes and trains a Ridge regression model and calculates MSE for both training and test data.'''
    
    # Make design matrix and split into training and test data
    X = PolynomialFeatures(pol_degree, include_bias = False).fit_transform(x) # without intercept
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3) # random_state gives same partition across multiple function calls
    
    # Scale data by subtracting mean
    X_scaler = np.mean(X_train, axis = 0)
    y_scaler = np.mean(y_train)
    X_train_scaled = X_train - X_scaler
    y_train_scaled = y_train - y_scaler
    X_test_scaled = X_test - X_scaler

    beta = [0] * len(lmbda)
    mse_train = np.zeros_like(lmbda)
    mse_test = np.zeros_like(lmbda)
    r2_train = np.zeros_like(lmbda)
    r2_test = np.zeros_like(lmbda)

    for i in range(len(lmbda)):
        # Fit parametres
        beta[i] = beta_ridge(X_train_scaled, y_train_scaled, lmbda[i])

        # Make predictions
        y_train_pred = X_train_scaled @ beta[i] + y_scaler
        y_test_pred = X_test_scaled @ beta[i] + y_scaler

        # Calculate MSE and R^2 for both training and test data
        mse_train[i] = mse_own(y_train_pred, y_train)
        mse_test[i] = mse_own(y_test_pred, y_test)
        r2_train[i] = r2_own(y_train_pred, y_train)
        r2_test[i] = r2_own(y_test_pred, y_test)

    return beta, mse_train, mse_test, r2_train, r2_test

def plot_ols(train_results, test_results, ylabel, name):
    '''Plots either MSE or R2 score for train and test data from OLS and saves to file.'''
    plt.figure(figsize = (6,4))
    plt.plot(range(1, len(train_results)+1), train_results, label = "Training data")
    plt.plot(range(1, len(train_results)+1), test_results, label = "Test data")
    plt.xlabel("Polynomial degree")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"plots/{name}.pdf")

def plot_ridge(train_results, test_results, ylabel, name):
    '''Plots either MSE or R2 score for train and test data from Ridge regression and saves to file.'''
    plt.figure(figsize = (6,12))
    for i in range(n_deg_max): # one subplot for each polynomial degree
        plt.subplot(n_deg_max, 1, i+1)

        plt.semilogx(lmbda, train_results[i], label = "Training data")
        plt.semilogx(lmbda, test_results[i], label = "Test data")

        plt.title(f"Polynomial of order {i+1}")
        plt.xlabel("$\lambda$")
        plt.ylabel(ylabel)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{name}.pdf")

# Set up dataset
n = 11 # number of points along one axis, total number of points will be n^2
start_value = 0
stop_value = 1
x = np.sort(np.random.rand(n, 1), axis = 0)
y = np.sort(np.random.rand(n, 1), axis = 0)
x_, y_ = np.meshgrid(x, y)
xy = np.stack((np.ravel(x_),np.ravel(y_)), axis = -1) # formatting needed to set up the design matrix
# z = add_noise(FrankeFunction(x_, y_))
z = FrankeFunction(x_, y_)

# Plot Franke function
fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
ax.plot_surface(x_, y_, z, cmap = cm.coolwarm)
fig.savefig("plots/franke.pdf")

n_deg_max = 5 # max polynomial degree

# Initialise dictionaries to store results
keys = ["beta", "mse_train", "mse_test", "r2_train", "r2_test"]
ols = dict.fromkeys(keys)
ridge = dict.fromkeys(keys)
lasso = dict.fromkeys(keys)
for key in ols.keys():
    ols[key] = [0]*n_deg_max
    ridge[key] = [0]*n_deg_max
    lasso[key] = [0]*n_deg_max

# Calculate OLS for polynomials of degree 1 to n_deg_max
for i in range(n_deg_max):
    ols_results = fit_predict_ols(xy, z.flatten(), i+1)
    ols["beta"][i] = ols_results[0]
    ols["mse_train"][i] = ols_results[1]
    ols["mse_test"][i] = ols_results[2]
    ols["r2_train"][i] = ols_results[3]
    ols["r2_test"][i] = ols_results[4]

# Plot result
plot_ols(ols["mse_train"], ols["mse_test"], "Mean Squared Error", "mse_ols")
plot_ols(ols["r2_train"], ols["r2_test"], f"$R^2$", "r2_ols")

lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0] # lambdas to try with Ridge regression

# Calculate Ridge regression for polynomials of degree 1 to n_deg_max
for i in range(n_deg_max):
    ridge_results = fit_predict_ridge(xy, z.flatten(), i+1, lmbda)
    ridge["beta"][i] = ridge_results[0]
    ridge["mse_train"][i] = ridge_results[1]
    ridge["mse_test"][i] = ridge_results[2]
    ridge["r2_train"][i] = ridge_results[3]
    ridge["r2_test"][i] = ridge_results[4]

# Plot result
plot_ridge(ridge["mse_train"], ridge["mse_test"], "Mean Squared Error", "mse_ridge")
plot_ridge(ridge["r2_train"], ridge["r2_test"], f"$R^2$", "r2_ridge")