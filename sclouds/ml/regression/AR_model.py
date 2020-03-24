import numpy as np
import scipy as sp

class AR_model:
""" Autoregressive models used in this thesis. Inspired by sklearn"""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        # bias is first column
        self.weights = None
       

    def fit(X, y):
        self.weights = sp.linalg.inv(X.T @ X)@ X.T @ y
        return

    def predict(X):
        """ Make prediction either pixelbased or full domain """
        return X@self.weights 

    
