import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    mu = np.array([[5],
                   [6],
                   [7],
                   [8],
                   ])
    S = np.array(
          [[+3.01602775,  1.02746769, -3.60224613, -2.08792829],
           [+1.02746769,  5.65146472, -3.98616664,  0.48723704],
           [-3.60224613, -3.98616664, 13.04508284, -1.59255406],
           [-2.08792829,  0.48723704, -1.59255406,  8.28742469]])  # noqa

    n = 1000

    d, U = np.linalg.eig(S)
    L = np.diagflat(d)
    A = np.dot(U, np.sqrt(L))
    X = np.random.randn(4, n)
    Y = np.dot(A, X) + np.tile(mu, n)
    print(Y)
    df = pd.DataFrame(Y.T, columns=['x1', 'x2', 'x3', 'x4'])
    sns.pairplot(df)
    plt.show()
    # plt.plot(X[0], X[1], '+b')
    # plt.plot(Y[0], Y[1], '+b')
    # plt.show()



if __name__ == '__main__':
    main()
