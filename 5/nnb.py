from numpy import tile, sum, argmin
import numpy as np
from scipy.stats import mode


class NNb:
    def __init__(self, X, c):
        self.n, self.N = X.shape
        self.X = X
        self.c = c

    def classify(self, x):
        d = self.X - tile(x.reshape(self.n, 1), self.N)
        dsq = sum(d * d, 0)
        minindex = argmin(dsq)
        return self.c[minindex]


class kNNb:
    def __init__(self, X, c):
        self.n, self.N = X.shape
        self.X = X
        self.c = c

    def classify(self, x, k):
        d = self.X - tile(x.reshape(self.n, 1), self.N)
        dsq = sum(d * d, 0)
        best_k_i = np.argsort(dsq, axis=0)[:k]
        best_k = self.c[best_k_i]
        return mode(best_k)[0][0]
