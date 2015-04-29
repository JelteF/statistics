import numpy as np


N = 100000


def main():
    mu = np.array([[5],
                   [6],
                   [7],
                   [8],
                   ])
    S = np.array([[3.01602775, 1.02746769, -3.60224613, -2.08792829],
                  [1.02746769, 5.65146472, -3.98616664, 0.48723704],
                  [-3.60224613, -3.98616664, 13.04508284, -1.59255406],
                  [-2.08792829, 0.48723704, -1.59255406, 8.28742469],
                  ])
    mu_est, S_est = estimate(mu, S)

    print('mean estimation:', mu_est)
    print('S estimation:')
    print(S_est)

    print()

    print('mean % difference:', (np.diagonal(mu / mu_est) - 1) * 100)
    print('S % difference:')
    print(((S / S_est) - 1) * 100)


def estimate(mu, S):
    d, U = np.linalg.eig(S)
    L = np.diagflat(d)
    A = np.dot(U, np.sqrt(L))
    X = np.random.randn(4, N)
    Y = np.dot(A, X) + np.tile(mu, N)

    Ybar = np.mean(Y, 1)
    Yzm = Y - np.tile(Ybar[:, None], N)
    Sest = np.dot(Yzm, Yzm.T) / (N-1)

    return Ybar, Sest


if __name__ == '__main__':
    main()
