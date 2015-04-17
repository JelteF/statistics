import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns  # noqa
from itertools import islice

from ibm_rng import ibm_rng

from pylatex import Plt


N = 1000


def main():
    dist_to_tex('np_random.tex', 'Numpy RNG', *get_random_pairs(N).T)

    ibm_gen = ibm_rng(1)
    dist_to_tex('ibm_random.tex', 'IBM RNG',
                take_samples(ibm_gen), take_samples(ibm_gen))


def take_samples(gen, n=N):
    return np.array(list(islice(gen, n)))


def dist_to_tex(filename, caption, x, y):
    plot_distribution(x, y, show=False)
    with open(filename, 'w') as f:
        plot = Plt(position='htbp')
        plot.add_plot(plt)
        plot.add_caption(caption)

        plot.dump(f)


def plot_distribution(x, y, show=True):
    sns.jointplot(x, y, stat_func=None)
    if show:
        plt.show()


def get_random_pairs(n):
    return rd.rand(n, 2)


if __name__ == '__main__':
    main()
