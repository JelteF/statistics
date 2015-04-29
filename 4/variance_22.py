import numpy as np


def main():
    mu = np.array([[5],
                   [6],
                   [7],
                   [8],
                   ])
    S = np.array([[2, 0, 0, 0],
                  [0, 2, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 2],
                  ])

    d, U = np.linalg.eig(S)
    L = np.diagflat(d)
    print(L)
    A = np.dot(U, np.sqrt(L))
    X = np.random.randn(4, 1000)
    Y = np.dot(A, X) + np.tile(mu, 1000)

    Ybar = np.mean(Y, 1)
    Yzm = Y - np.tile(Ybar[:, None], 1000)
    Sest = np.dot(Yzm, Yzm.T) / 999

    print('Ybar', Ybar)
    print('Sest', Sest)


if __name__ == '__main__':
    main()
