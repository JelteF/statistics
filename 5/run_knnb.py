from numpy import arange, loadtxt, transpose,\
    zeros, sum, array, logical_and
import matplotlib.pyplot as plt
import numpy.random as rd
from nnb import kNNb
import numpy as np


def cnvt(s):
    tab = {'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0}
    s = s.decode()
    if s in tab:
        return tab[s]
    else:
        return -1.0

XC = loadtxt('data/iris.data', delimiter=',', dtype=float,
             converters={4: cnvt})


def main():
    seed = rd.randint(10000)
    test_k = range(1, 10, 2)
    n = 400
    accuracies = np.zeros((n, len(test_k)))
    for i in range(n):
        for j, k in enumerate(test_k):
            accuracies[i, j] = do_knnb(k, seed, plot=False, print_=False)
        seed += 1

    mean_accuracies = np.mean(accuracies, axis=0)

    print('Acuracy for every k\'s from 1 to 9')
    for k, a in zip(test_k, mean_accuracies):
        print(k, a, sep=': ')


def do_knnb(k, seed=None, plot=True, print_=True):
    if seed is not None:
        rd.seed(seed)
    ind = arange(150)  # indices into the dataset
    ind = rd.permutation(ind)  # random permutation
    L = ind[0:90]  # learning set indices
    T = ind[90:]  # test set indices

    # Learning Set
    X = transpose(XC[L, 0:-1])
    nnc = kNNb(X, XC[L, -1])

    # Classification of Test Set
    c = zeros(len(T))
    for i in arange(len(T)):
        c[i] = nnc.classify(XC[T[i], 0:4], k)

    # Confusion Matrix
    CM = zeros((3, 3))
    for i in range(3):
        for j in range(3):
            CM[i, j] = sum(logical_and(XC[T, -1] == (i+1), c == (j+1)))
    if print_:
        print(k)
        print(CM)

    if plot:
        plot_stuff(T, c)

    return np.sum(c == XC[T, -1]) / len(XC[T])


def plot_stuff(T, c):
    # Plot Test Set
    plt.figure(1)

    color = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, 4*i + j + 1)
            if i == j:
                continue
            plt.scatter(XC[T, i], XC[T, j], s=100, marker='s',
                        edgecolor=color[XC[T, 4].astype(int)-1],
                        facecolor=[[1, 1, 1]] * len(T))
            plt.scatter(XC[T, i], XC[T, j], s=30, marker='+',
                        edgecolor=color[c.astype(int)-1])

    plt.show()

if __name__ == '__main__':
    main()
