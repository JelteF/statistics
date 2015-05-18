from numpy import tile, sum, argmin


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
