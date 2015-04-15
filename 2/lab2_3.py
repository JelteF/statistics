from scipy.stats import norm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  # noqa
# from pylatex import Plt


def main():
    rv = norm()
    x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)

    pdf = rv.pdf(x)
    cdf = rv.cdf(x)

    r = rv.rvs(size=1000)

    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, pdf, label=r'$\phi\text{(pdf)}$')
    ax.plot(x, cdf, label=r'$\Phi\text{(cdf)}$')
    ax.hist(r, bins=30, normed=True, histtype='stepfilled', alpha=0.2)
    ax.legend(loc='best', frameon=True)
    plt.show()

    # plot = Plt(position='htbp')
    # plot.add_plot(plt)

    # with open('plot2_3.tex', 'w') as f:
    #   plot.dump(f)


if __name__ == '__main__':
    main()
