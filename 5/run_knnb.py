from numpy import arange, loadtxt, transpose,\
    zeros, sum, array, logical_and
import matplotlib.pyplot as plt
from numpy.random import permutation
from nnb import NNb


def cnvt(s):
    tab = {'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0}
    s = s.decode()
    if s in tab:
        return tab[s]
    else:
        return -1.0

XC = loadtxt('data/iris.data', delimiter=',', dtype=float,
             converters={4: cnvt})

ind = arange(150)  # indices into the dataset
ind = permutation(ind)  # random permutation
L = ind[0:90]  # learning set indices
T = ind[90:]  # test set indices

# Learning Set
X = transpose(XC[L, 0:4])
nnc = NNb(X, XC[L, -1])

# Classification of Test Set
c = zeros(len(T))
for i in arange(len(T)):
    c[i] = nnc.classify(XC[T[i], 0:4])

# Confusion Matrix
CM = zeros((3, 3))
for i in range(3):
    for j in range(3):
        CM[i, j] = sum(logical_and(XC[T, 4] == (i+1), c == (j+1)))
print(CM)

# Plot Test Set
plt.figure(1)

color = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 4*i + j + 1)
        if i == j:
            continue
        print(XC[T, 4].astype(int))
        plt.scatter(XC[T, i], XC[T, j], s=100, marker='s',
                    edgecolor=color[XC[T, 4].astype(int)-1],
                    facecolor=[[1, 1, 1]] * len(T))
        plt.scatter(XC[T, i], XC[T, j], s=30, marker='+',
                    edgecolor=color[c.astype(int)-1])
plt.show()
