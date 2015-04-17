import math
import numpy as np
import matplotlib.pyplot as plt
from uniform import take_samples
from scipy.stats import expon
from pylatex import Plt


N = 10000


def expo(lmbda):
    def cdf_inv(p):
        return (-math.log(1-p, math.e))/lmbda

    while True:
        yield cdf_inv(np.random.rand())


def main():
    gen = expo(1.0)
    y = take_samples(gen, N)

    rv = expon()
    x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)
    pdf = rv.pdf(x)

    plt.plot(x, pdf, label='pdf')
    plt.hist(y, normed=True, bins=30, label='Trekkingen')
    plt.legend(loc='best', frameon=True)

    with open('expo.tex', 'w') as f:
        plot = Plt(position='htbp')
        plot.add_plot(plt)
        plot.add_caption('Exponentiële verdeling')
        plot.dump(f)

    # plt.show()

    lmbda = 1.0 / np.mean(y)
    print('Geschatte lambda: %f' % (lmbda))


if __name__ == '__main__':
    main()
