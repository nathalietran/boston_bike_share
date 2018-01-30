import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import BayesianRidge

# Remember : there are 171 stations


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = []
        for i in range(171):
            self.reg.append(BayesianRidge())

    def fit(self, X, y):
        for i in range(171):
            self.reg[i].fit(X[i::171], y[i::171].ravel())

    def predict(self, X):
        y_tmp = []
        for i in range(171):
            y_tmp.append(self.reg[i].predict(X[i::171]))
        y_tmp = np.array(y_tmp)
        y_final = []
        for j in range(len(y_tmp[0])):
            for k in range(171):
                y_final.append(max(0, np.int(y_tmp[k][j])))
        return np.array(y_final)[:, np.newaxis]
