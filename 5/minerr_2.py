from numpy import arange, loadtxt, \
    zeros, sum, array, logical_and
import matplotlib.pyplot as plt
import numpy.random as rd
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def cnvt(s):
    tab = {'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0}
    s = s.decode()
    if s in tab:
        return tab[s]
    else:
        return -1.0

DATA = loadtxt('data/iris.data', delimiter=',', dtype=float,
               converters={4: cnvt})


def main():
    seed = rd.randint(10000)
    n = 400
    accuracies = np.zeros((n, ))
    for i in range(n):
        accuracies[i] = do_minerr(seed, plot=False, print_=False)
        seed += 1

    mean_accuracy = np.mean(accuracies)

    print('The accuracy is: ', mean_accuracy * 100, '%', sep='')


class MinError():
    def __init__(self, X):
        X = pd.DataFrame(X)
        self.pdfs = {}
        self.class_chances = {}
        for name, g in X.groupby(X.columns[-1]):
            data = g.as_matrix()[:, :-1]
            mean = data.mean(axis=0)
            cov = np.cov(data.T)
            self.pdfs[name] = multivariate_normal(mean=mean, cov=cov).pdf
            self.class_chances[name] = len(g) / len(X)

    def classify(self, x):
        best_class = None
        best_chance = 0
        for cls, pdf in self.pdfs.items():
            chance = pdf(x) * self.class_chances[cls]
            if chance > best_chance:
                best_chance = chance
                best_class = cls

        return best_class


def do_minerr(seed=None, plot=True, print_=True):
    if seed is not None:
        rd.seed(seed)
    ind = arange(150)  # indices into the dataset
    ind = rd.permutation(ind)  # random permutation
    L = ind[0:90]  # learning set indices
    T = ind[90:]  # test set indices

    # Learning Set
    X = DATA[L, :]
    classifier = MinError(X)

    # Classification of Test Set
    c = zeros(len(T))
    for i in arange(len(T)):
        c[i] = classifier.classify(DATA[T[i], 0:4])

    # Confusion Matrix
    CM = zeros((3, 3))
    for i in range(3):
        for j in range(3):
            CM[i, j] = sum(logical_and(DATA[T, -1] == (i+1), c == (j+1)))
    if print_:
        print(CM)

    if plot:
        plot_stuff(T, c)

    return np.sum(c == DATA[T, -1]) / len(DATA[T])


def plot_stuff(T, c):
    # Plot Test Set
    plt.figure(1)

    color = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, 4*i + j + 1)
            if i == j:
                continue
            plt.scatter(DATA[T, i], DATA[T, j], s=100, marker='s',
                        edgecolor=color[DATA[T, 4].astype(int)-1],
                        facecolor=[[1, 1, 1]] * len(T))
            plt.scatter(DATA[T, i], DATA[T, j], s=30, marker='+',
                        edgecolor=color[c.astype(int)-1])

    plt.show()

if __name__ == '__main__':
    main()
