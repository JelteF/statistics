import numpy as np
from scipy import stats
import matplotlib.pyplot as pl


P = [0.3, 0.7]
mu = [4, 7]
sigma = [1, 2]


def p_xc(x, C):
    P_C = P[C-1]
    mu_C = mu[C-1]
    sigma_C = sigma[C-1]

    p_x_c = stats.norm(loc=mu_C, scale=sigma_C)

    return p_x_c.pdf(x) * P_C


def main():
    n = 1000
    X = np.linspace(-4, 15, n)

    p_xc1 = p_xc(X, 1)
    p_xc2 = p_xc(X, 2)

    P_cx = np.zeros((n, ))
    for i, (x, y1, y2) in enumerate(zip(X, p_xc1, p_xc2)):
        px = y1 + y2

        P_cx[i] = y1 / px

    pl.plot(X, p_xc1, 'b')
    pl.plot(X, p_xc1/P[0], 'b')
    pl.plot(X, p_xc2, 'r')
    pl.plot(X, p_xc2/P[1], 'r')
    pl.plot(X, P_cx, 'b--')
    pl.plot(X, 1-P_cx, 'r--')
    pl.show()


if __name__ == '__main__':
    main()
