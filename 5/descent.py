import numpy as np
import numpy.random as rd
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

th1 = 2
th2 = .5
x = np.linspace(0, 10, 101)
y = th1 * np.sin(th2 * x) + 0.3 * rd.randn(*x.shape)


def J(th):
    return np.sum((y - th[0] * np.sin(th[1] * x))**2)


def derivative1(eth1, eth2):
    return np.sum(2 * np.sin(eth2 * x) * derivative_common(eth1, eth2))


def derivative2(eth1, eth2):
    return np.sum(2 * eth1 * x * np.cos(eth2 * x) *
                  derivative_common(eth1, eth2))


def derivative_common(eth1, eth2):
    return -y + eth1 * np.sin(eth2 * x)


def Jac(th):
    print(th)
    d = np.array([derivative1(*th), derivative2(*th)])
    print(d)
    return d


res = minimize(J, (10, .6), jac=Jac, method='CG')
print(res)
eth1, eth2 = res.x
plt.plot(x, y, 'xb')
plt.plot(x, eth1 * np.sin(eth2 * x), 'b')
plt.plot(x, th1 * np.sin(th2 * x), 'g')
plt.show('figures/linregression.png')
