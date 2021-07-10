import numpy as np
from sklearn.metrics import mean_squared_error

def get_precision(X:list, Y:list):
    _intersection = set(X).intersection(Y)
    return len(_intersection) / len(Y)

def get_recall(X:list, Y:list):
    _intersection = set(X).intersection(Y)
    return len(_intersection) / len(X)

def get_rmse(X, X_hat):
    return np.sqrt(mean_squared_error(X, X_hat))
def get_rmse(X, X_hat):
    return np.sqrt(mean_squared_error(X, X_hat))