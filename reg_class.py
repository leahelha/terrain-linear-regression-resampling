import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib import cm

class regression_class:
    '''Does OLS, Ridge and Lasso regression with a polynomial model of degree up to n_deg_max.'''

    def __init__(self, x, y, n_deg_max, lmbda):
        self.x = x
        self.y = y
        self.n_deg_max = n_deg_max
        self.lmbda = lmbda

        # Initialise dictionaries to store results
        keys = ["beta", "mse_train", "mse_test", "r2_train", "r2_test"]
        self.ols = dict.fromkeys(keys)
        self.ridge = dict.fromkeys(keys)
        self.lasso = dict.fromkeys(keys)
        for key in self.ols.keys():
            self.ols[key] = [0]*n_deg_max
            self.ridge[key] = [0]*n_deg_max
            self.lasso[key] = [0]*n_deg_max
        
        self.make_design_matrix()
        self.scale_design_matrix()
    
    def ols_regression(self):
        '''Calculates OLS for polynomials of degree 1 to n_deg_max.'''
        for i in range(self.n_deg_max):
            ols_results = self.fit_predict_ols(i+1)
            self.ols["beta"][i] = ols_results[0]
            self.ols["mse_train"][i] = ols_results[1]
            self.ols["mse_test"][i] = ols_results[2]
            self.ols["r2_train"][i] = ols_results[3]
            self.ols["r2_test"][i] = ols_results[4]

    def ridge_regression(self):
        '''Calculates Ridge regression for polynomials of degree 1 to n_deg_max.'''
        for i in range(self.n_deg_max):
            ridge_results = self.fit_predict_ridge(i+1)
            self.ridge["beta"][i] = ridge_results[0]
            self.ridge["mse_train"][i] = ridge_results[1]
            self.ridge["mse_test"][i] = ridge_results[2]
            self.ridge["r2_train"][i] = ridge_results[3]
            self.ridge["r2_test"][i] = ridge_results[4]
    
    def lasso_regression(self):
        '''Calculates Lasso regression for polynomials of degree 1 to n_deg_max.'''
        for i in range(self.n_deg_max):
            lasso_results = self.fit_predict_lasso(i+1)
            self.lasso["beta"][i] = lasso_results[0]
            self.lasso["mse_train"][i] = lasso_results[1]
            self.lasso["mse_test"][i] = lasso_results[2]
            self.lasso["r2_train"][i] = lasso_results[3]
            self.lasso["r2_test"][i] = lasso_results[4]

    def plot_ols_results(self):
        '''Plots MSE, R2 score and beta values for OLS regression and saves to file.'''
        self.plot_ols(self.ols["mse_train"], self.ols["mse_test"], "Mean Squared Error", "mse_ols")
        self.plot_ols(self.ols["r2_train"], self.ols["r2_test"], f"$R^2$", "r2_ols")
        self.plot_beta_ols(self.ols["beta"], "beta_ols")
    
    def plot_ridge_results(self):
        '''Plots MSE, R2 score and beta values for Ridge regression and saves to file.'''
        self.plot_ridge_or_lasso(self.ridge["mse_train"], self.ridge["mse_test"], "Mean Squared Error", "mse_ridge")
        self.plot_ridge_or_lasso(self.ridge["r2_train"], self.ridge["r2_test"], f"$R^2$", "r2_ridge")
        self.plot_beta_ridge_or_lasso(self.ridge["beta"], self.lmbda, "beta_ridge")

    def plot_lasso_results(self):
        '''Plots MSE, R2 score and beta values for Lasso regression and saves to file.'''
        self.plot_ridge_or_lasso(self.lasso["mse_train"], self.lasso["mse_test"], "Mean Squared Error", "mse_lasso")
        self.plot_ridge_or_lasso(self.lasso["r2_train"], self.lasso["r2_test"], f"$R^2$", "r2_lasso")
        self.plot_beta_ridge_or_lasso(self.lasso["beta"], self.lmbda, "beta_lasso")

    def plot_ols(self, train_results, test_results, ylabel, name):
        '''Plots either MSE or R2 score for train and test data from OLS and saves to file.'''
        plt.figure(figsize = (6,4))
        plt.plot(range(1, len(train_results)+1), train_results, label = "Training data")
        plt.plot(range(1, len(train_results)+1), test_results, label = "Test data")
        plt.xlabel("Polynomial degree")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(f"plots/{name}.pdf")
    
    def plot_ridge_or_lasso(self, train_results, test_results, ylabel, name):
        '''Plots either MSE or R2 score for train and test data from Ridge or Lasso regression and saves to file.'''
        plt.figure(figsize = (6,12))
        for i in range(self.n_deg_max): # one subplot for each polynomial degree
            plt.subplot(self.n_deg_max, 1, i+1)

            plt.semilogx(self.lmbda, train_results[i], label = "Training data")
            plt.semilogx(self.lmbda, test_results[i], label = "Test data")

            plt.title(f"Polynomial of order {i+1}")
            plt.xlabel("$\lambda$")
            plt.ylabel(ylabel)
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{name}.pdf")

    def plot_beta_ols(self, beta, name):
        '''Plots beta values with standard deviation from OLS regression.'''
        plt.figure(figsize = (15,5))
        labels = ["$x$", "$y$", "$x^2$", "$xy$", "$y^2$", "$x^3$", "$x^2y$", "$xy^2$", "$y^3$", "$x^4$", "$x^3y$",
                "$x^2y^2$", "$xy^3$", "$y^4$", "$x^5$", "$x^4y$", "$x^3y^2$", "$x^2y^3$", "$xy^4$", "$y^5$"]
        for i in range(len(beta)):
            plt.subplot(1, len(beta), i+1)
            plt.bar(labels[:len(beta[i])], beta[i])
        plt.savefig(f"plots/{name}.pdf")

    def plot_beta_ridge_or_lasso(self, beta, lmbda, name):
        '''Plots beta values with standard deviation from Ridge or Lasso regression.'''
        labels = ["$x$", "$y$", "$x^2$", "$xy$", "$y^2$", "$x^3$", "$x^2y$", "$xy^2$", "$y^3$", "$x^4$", "$x^3y$",
                "$x^2y^2$", "$xy^3$", "$y^4$", "$x^5$", "$x^4y$", "$x^3y^2$", "$x^2y^3$", "$xy^4$", "$y^5$"]
        for i in range(len(lmbda)):
            plt.figure(figsize = (15,5))
            for j in range(len(beta)):
                plt.subplot(1, len(beta), j+1)
                plt.bar(labels[:len(beta[j][i])], beta[j][i])
            plt.savefig(f"plots/{name}_{lmbda[i]}.pdf")

    def fit_predict_ols(self, pol_degree):
        '''For a given polynomial order, makes and trains an OLS model and calculates MSE for both training and test data.'''
        
        # Pick out relevant part of design matrix
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X_train_scaled = self.X_train_scaled[:, 0:N]
        X_test_scaled = self.X_test_scaled[:, 0:N]
        
        # Fit parametres
        beta = self.beta_ols(X_train_scaled, self.y_train_scaled)

        # Make predictions
        y_train_pred = X_train_scaled @ beta + self.y_scaler
        y_test_pred = X_test_scaled @ beta + self.y_scaler

        # Calculate MSE and R^2 for both training and test data
        mse_train = self.mse_own(y_train_pred, self.y_train)
        mse_test = self.mse_own(y_test_pred, self.y_test)
        r2_train = self.r2_own(y_train_pred, self.y_train)
        r2_test = self.r2_own(y_test_pred, self.y_test)

        return beta, mse_train, mse_test, r2_train, r2_test
    
    def fit_predict_ridge(self, pol_degree):
        '''For a given polynomial order, makes and trains a Ridge regression model and calculates MSE for both training and test data.'''
        # Pick out relevant part of design matrix
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X_train_scaled = self.X_train_scaled[:, 0:N]
        X_test_scaled = self.X_test_scaled[:, 0:N]

        beta = [0] * len(self.lmbda)
        mse_train = np.zeros_like(self.lmbda)
        mse_test = np.zeros_like(self.lmbda)
        r2_train = np.zeros_like(self.lmbda)
        r2_test = np.zeros_like(self.lmbda)

        for i in range(len(self.lmbda)):
            # Fit parametres
            beta[i] = self.beta_ridge(X_train_scaled, self.y_train_scaled, self.lmbda[i])

            # Make predictions
            y_train_pred = X_train_scaled @ beta[i] + self.y_scaler
            y_test_pred = X_test_scaled @ beta[i] + self.y_scaler

            # Calculate MSE and R^2 for both training and test data
            mse_train[i] = self.mse_own(y_train_pred, self.y_train)
            mse_test[i] = self.mse_own(y_test_pred, self.y_test)
            r2_train[i] = self.r2_own(y_train_pred, self.y_train)
            r2_test[i] = self.r2_own(y_test_pred, self.y_test)

        return beta, mse_train, mse_test, r2_train, r2_test
    
    def fit_predict_lasso(self, pol_degree):
        '''For a given polynomial order, makes and trains a Lasso regression model and calculates MSE for both training and test data.'''
        # Pick out relevant part of design matrix
        N = int((pol_degree+1)*(pol_degree+2)/2 - 1)
        X_train_scaled = self.X_train_scaled[:, 0:N]
        X_test_scaled = self.X_test_scaled[:, 0:N]

        beta = [0] * len(self.lmbda)
        mse_train = np.zeros_like(self.lmbda)
        mse_test = np.zeros_like(self.lmbda)
        r2_train = np.zeros_like(self.lmbda)
        r2_test = np.zeros_like(self.lmbda)

        for i in range(len(self.lmbda)):
            # Fit parametres
            model = Lasso(self.lmbda[i], max_iter = 5000, tol = 1e-2).fit(X_train_scaled, self.y_train_scaled)
            beta[i] = model.coef_

            # Make predictions
            y_train_pred = X_train_scaled @ beta[i] + self.y_scaler
            y_test_pred = X_test_scaled @ beta[i] + self.y_scaler

            # Calculate MSE and R^2 for both training and test data
            mse_train[i] = self.mse_own(y_train_pred, self.y_train)
            mse_test[i] = self.mse_own(y_test_pred, self.y_test)
            r2_train[i] = self.r2_own(y_train_pred, self.y_train)
            r2_test[i] = self.r2_own(y_test_pred, self.y_test)

        return beta, mse_train, mse_test, r2_train, r2_test

    def beta_ols(self, X, y):
        '''Given the design matrix X and the output y, calculates the coefficients beta using OLS.'''
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta

    def beta_ridge(self, X, y, lmbda):
        '''Given the design matrix X, the output y and the parameter lmbda, calculates the coefficients beta using OLS.'''
        n = np.shape(X)[1]
        beta = np.linalg.inv(X.T @ X + lmbda*np.eye(n)) @ X.T @ y
        return beta

    def mse_own(self, y_tilde, y):
        '''Calculates the mean square error of a prediction y_tilde.'''
        mse = 1/len(y) * np.sum((y-y_tilde)**2)
        return mse

    def r2_own(self, y_tilde, y):
        '''Calculates the R^2 score of a prediction y_tilde.'''
        a = np.sum((y-y_tilde)**2)
        b = np.sum((y-np.mean(y))**2)
        return 1 - a/b

    def make_design_matrix(self):
        '''Makes design matrix and splits into training and test data'''
        self.X = PolynomialFeatures(self.n_deg_max, include_bias = False).fit_transform(self.x) # without intercept
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 3) # random_state gives same partition across multiple function calls

    def scale_design_matrix(self):
        '''Scales data by subtracting mean'''
        self.X_scaler = np.mean(self.X_train, axis = 0)
        self.y_scaler = np.mean(self.y_train)
        self.X_train_scaled = self.X_train - self.X_scaler
        self.y_train_scaled = self.y_train - self.y_scaler
        self.X_test_scaled = self.X_test - self.X_scaler
        

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


def main():
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

    model = regression_class(xy, z.flatten(), n_deg_max, lmbda) # Kall det polynomial istedenfor

    # Do regression
    model.ols_regression()
    model.ridge_regression()
    model.lasso_regression()

    # Plot results
    model.plot_ols_results()
    model.plot_ridge_results()
    model.plot_lasso_results()

if __name__ == "__main__":
    main()