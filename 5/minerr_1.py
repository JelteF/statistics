import numpy as np
from scipy import stats
import matplotlib.pyplot as pl
import seaborn as sns  # noqa
from itertools import count


P = [0.3, 0.7]
mu = [4, 7]
sigma = [1, 1.5]
# The second sigma is 1.5 so it corresponds with the graph in the section
# 2.4.2, it didn't work with the sigma=2 mentioned there.


def p_xc(x, C):
    """Get pxc(x, C=C)"""
    mu_C = mu[C-1]
    sigma_C = sigma[C-1]

    p_x_c = stats.norm(loc=mu_C, scale=sigma_C)  # px|c(x|C=C)

    return p_x_c.pdf(x) * P[C-1]


def main():
    n = 100
    X = np.linspace(-4, 15, n)

    p_xc1 = p_xc(X, 1)
    p_xc2 = p_xc(X, 2)

    P_cx = np.zeros((n, ))
    for i, y1, y2 in zip(count(), p_xc1, p_xc2):
        px = y1 + y2  # px(x)

        P_cx[i] = y1 / px  # P(C=1|x)

    pl.plot(X, p_xc1, 'b')
    pl.plot(X, p_xc2, 'r')
    pl.plot(X, P_cx, 'b--')
    pl.plot(X, 1 - P_cx, 'r--')

    pl.show()


if __name__ == '__main__':
    main()
