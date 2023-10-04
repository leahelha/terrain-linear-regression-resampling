import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from matplotlib import cm
from pprint import pprint
from tqdm import trange

def FrankeFunction(x,y):
    '''Calculates the two-dimensional Franke's function.'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def kFold_linreg(x, y, degree, lin_model, lmbda=None):
    np.random.seed(1234)
    poly = PolynomialFeatures(degree, include_bias = True) # Skal være False hvis sentrerer
    if lmbda is None:
        model = lin_model(fit_intercept = True) # Skal være False hvis sentrerer
    else:
        model = lin_model(alpha = lmbda, fit_intercept = False)

    indicies = np.arange(len(x))
    np.random.shuffle(indicies)

    x_shuffled = x[indicies]
    y_shuffled = y[indicies]

    # Initialize a KFold instance:
    k = 5
    kfold = KFold(n_splits = k)
    scores_KFold = np.zeros(k)

    # Perform the cross-validation to estimate MSE:
    for i, (train_inds, test_inds) in enumerate(kfold.split(x_shuffled)):
        x_train = x_shuffled[train_inds]
        y_train = y_shuffled[train_inds]

        x_test = x_shuffled[test_inds]
        y_test = y_shuffled[test_inds]

        # Train: Centring and design matrix
        X_train = poly.fit_transform(x_train)
        # X_train_scalar = np.mean(X_train, axis = 0)
        # y_train_scalar = np.mean(y_train)

        # X_centred_train = X_train - X_train_scalar
        # y_centred_train = y_train - y_train_scalar

        # Test: Centring and design matrix
        X_test = poly.fit_transform(x_test)
        # X_test_scalar = np.mean(X_test, axis = 0)
        # y_test_scalar = np.mean(y_test)

        # X_centred_test = X_test - X_test_scalar
        # y_centred_test = y_test - y_test_scalar

        # Fitting on train data, and predicting on test data:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Scores: mse
        scores_KFold[i] = np.sum((y_pred - y_test)**2)/np.size(y_pred)


    scores_KFold_mean = np.mean(scores_KFold)

    poly = PolynomialFeatures(degree, include_bias = True) # Funker når det er True???
    linreg = LinearRegression(fit_intercept = False) # Når den er False så funker det???
    X = poly.fit_transform(x_shuffled)
    check_scores = -cross_val_score(linreg, X, y_shuffled, scoring="neg_mean_squared_error", cv=kfold)

    # x_centred = x - np.mean(x, axis = 0)
    # y_centred = y - np.mean(y)
    # X_centred = poly.fit_transform(x_centred)
    # check_scores = -cross_val_score(model, X_centred, y_centred, scoring = "neg_mean_squared_error", cv = kfold)
    return scores_KFold, check_scores
    # return scores_KFold_mean

def main():
    # Set up dataset
    n = 20 # number of points along one axis, total number of points will be n^2
    x = np.sort(np.random.rand(n, 1), axis = 0)
    y = np.sort(np.random.rand(n, 1), axis = 0)
    x_, y_ = np.meshgrid(x, y)
    xy = np.stack((np.ravel(x_),np.ravel(y_)), axis = -1) # formatting needed to set up the design matrix
    z = FrankeFunction(x_, y_)

    score, check = kFold_linreg(xy, z.flatten(), degree = 5, lin_model = LinearRegression)

    print("My code | Sklearn code")
    for s, c in zip(score, check):
        print(f"{s:0.7f}  |  {c:0.7f} | rel: {abs(s - c)/c:0.7E}")

    # nlmbda = 500
    # lmbda = np.logspace(-3, 5, nlmbda)
    # for i in trange(len(lmbda)):
    #     kFold_linreg(xy, z.flatten(), degree = 5, lin_model = Lasso, lmbda = lmbda[i])

    # print(kFold_linreg(xy, z.flatten(), degree = 5, lin_model = LinearRegression))


if __name__ == "__main__":
    main()