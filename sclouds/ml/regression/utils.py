import os

import numpy as np
import xarray as xr

def sigmoid(x):
    """ Computed the sigmoid transformation. Truncates real axis to  values in
    the range 0 and 1.

    Expression : np.exp(x)/(1 + np.exp(x)).

    Parameteres
    -------------------
    x : array-like
        Vector containing the values.

    Returnes
    --------------------
    s : array-like
        The sigmoid transform of x
    """
    s = np.exp(x)/(1 + np.exp(x))
    return s

def inverse_sigmoid(x):
    """Also known as the logit function. Expression np.log(x/(1-x).
    Use to transform the response to be in the range (-inf, +inf).

    Parameteres
    -------------------
    x : array-like
        Vector containing the values.

    Returnes
    --------------------
    _ : array-like
        The inverse sigmoid transform of x
    """
    return np.log(x/(1-x))

def mean_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error score metric.

    Parameteres
    ------------------
    y_true : array-like
        Actual vales of y.
    y_pred : array-like
        Predicted values of y.

    Returns
    -------------------
    mse : float
        mean squared error
    """
    mse = np.square(np.subtract(y_true, y_pred)).mean(axis = 0)
    return mse


def accumulated_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error score metric.

    Parameteres
    ----------------
    y_true : array-like
        Actual vales of y.
    y_pred : array-like
        Predicted values of y.

    Returns
    ----------------
    ase : float
        Accumulated squared error between y_true and y_pred.
    """
    ase = np.square(np.subtract(y_true, y_pred)).sum(axis = 0)
    return ase


def r2_score(y_true, y_pred):
    """ Computes the R2 score score metric.

    Parameteres
    ---------------------------
    y_true : array-like
        Actual vales of y.
    y_pred : array-like
        Predicted values of y.

    Returns
    ----------------------------
    r2 : float
         Coefficient of determination.

    Notes
    -----------
    Describes variation of data captured by the model.
    """
    numerator   = np.square(np.subtract(y_true, y_pred)).sum(axis=0)
    denominator = np.square(np.subtract(y_true, np.average(y_true))).sum(axis=0)
    val = numerator/denominator
    return 1 - val


def fit_pixel(X, y):
    """Traines one pixel of the grid.

    Parameteres
    -----------------

    Returns
    -------------
    coeffs :
    """
    from scipy.linalg import pinv
    coeffs = np.dot(pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return coeffs

def predict_pixel(X, coeffs):
    """Make prediction of one pixel. Return"""
    return np.dot(X, coeffs)
