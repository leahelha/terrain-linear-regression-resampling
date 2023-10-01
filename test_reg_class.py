import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from matplotlib import cm
import reg_class

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
lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0] # lambdas to try with Ridge regression

model = reg_class.regression_class(xy, z.flatten(), n_deg_max, lmbda)

# Do regression with our own code
model.ols_regression()
model.ridge_regression()

beta_ols = [0]*n_deg_max
beta_ridge = [0]*n_deg_max

X_train_scaled = model.X_train_scaled
y_train_scaled = (model.y_train_scaled).reshape(-1,1)

# Do regression with scikit-learn
for pol_degree in range(1,n_deg_max+1):
    N = int((pol_degree+1)*(pol_degree+2)/2 - 1)

    X_train_scaled_N = X_train_scaled[:, 0:N]

    m_ols = LinearRegression().fit(X_train_scaled_N, y_train_scaled)
    beta_ols[pol_degree-1] = m_ols.coef_[0]

    beta_lmbda = []
    for i in range(len(lmbda)):
        m_ridge = Ridge(alpha = lmbda[i]).fit(X_train_scaled_N, y_train_scaled)
        beta_lmbda.append(m_ridge.coef_[0])
    beta_ridge[pol_degree-1] = beta_lmbda

# Compare
print("--- TEST OLS ---")
beta_test = True
tol = 1e-5
for i in range(len(beta_ols)):
    for j in range(len(beta_ols[i])):
        if (np.abs(beta_ols[i][j] - model.ols["beta"][i][j]) > tol):
            beta_test = False
print(beta_test)

print("--- TEST RIDGE ---")
beta_test = True
tol = 1e-5
for i in range(len(beta_ridge)):
    for j in range(len(beta_ridge[i])):
        for k in range(len(beta_ridge[i][j])):
            if (np.abs(beta_ridge[i][j][k] - model.ridge["beta"][i][j][k]) > tol):
                beta_test = False
print(beta_test)