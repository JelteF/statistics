from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # noqa
from pprint import pprint  # noqa

from math import pi as π
from math import exp, sqrt
from collections import defaultdict
from operator import itemgetter, attrgetter

sex_dict = {'M': 'Man', 'F': 'Vrouw'}


def main():
    data = pd.read_csv('biometrie2014.csv')
    print(data)
    plot_dists(data)
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for i in range(len(data)):
        dists = get_distributions(data.drop(i))
        probs = defaultdict(lambda: 0.5)
        test_subject = data.iloc[i]
        for sex, cols in dists.items():
            for col, dist in cols.items():
                probs[sex] *= dist.pdf(test_subject[col])
        actual_sex = test_subject['Man/Vrouw']
        probable_sex = max(probs.items(), key=itemgetter(1))[0]
        confusion_matrix[probable_sex][actual_sex] += 1
    pprint(confusion_matrix)


def plot_dists(data):
    inside_out_dists = get_distributions(data)

    dists = defaultdict(dict)
    for k, v in inside_out_dists.items():
        for k2, pdf in v.items():
            dists[k2][k] = pdf

    for col_name, sexes in dists.items():
        print(sexes.values())
        lowerbound = min(map(attrgetter('lowerbound'), sexes.values()))
        upperbound = max(map(attrgetter('upperbound'), sexes.values()))
        print(lowerbound)
        x = np.linspace(lowerbound, upperbound, 10000)
        for sex, dist in sexes.items():
            plt.plot(x, np.array(list(map(dist.pdf, x))), label=sex_dict[sex])

        ax = plt.gca()
        ax.legend(loc='best', frameon=False)
        ax.set_title('Kansdichteid van ' + col_name.lower())
        ax.set_ylabel('Kansdichtheid')
        ax.set_xlabel(col_name)
        plt.show()


def get_distributions(data):
    dists = {}
    for name, group in data.groupby('Man/Vrouw'):
        d = {}
        dists[name] = d
        for µ, σ, col_name in zip(group.mean(), group.std(),
                                  group.columns.values[1:]):
            d[col_name] = NormalDistribution(µ, σ, col_name)
    return dists


class NormalDistribution:
    boundary_range = 5

    def __init__(self, µ, σ, name=''):
        self.µ = µ
        self.σ = σ
        self.lowerbound = µ - self.boundary_range * σ
        self.upperbound = µ + self.boundary_range * σ
        self.name = name

    def pdf(self, x):
        return 1 / (self.σ * sqrt(2 * π)) * \
            exp(-((x - self.μ)**2) / (2 * self.σ**2))

if __name__ == '__main__':
    main()
