import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path


n = 256
m = 25


def compare(img, pos, mean_vec, d, U, k, show=False):
    x, y = pos

    detail = img[x:x+m, y:y+m].reshape((m*m, 1))

    det_zm = detail - mean_vec
    y_zm = U.T.dot(det_zm)

    y_zm_k = y_zm[:k]
    det_zm_k = U[:, :k].dot(y_zm_k)
    det_k = det_zm_k + mean_vec

    if show:
        plt.subplot(2, 1, 1)
        plt.imshow(detail.reshape((m, m)), cmap=cm.Greys_r)

        plt.subplot(2, 1, 2)
        plt.imshow(det_k.reshape((m, m)), cmap=cm.Greys_r)

        plt.show()

    return det_k


def sorted_eig(mat):
    d, U = np.linalg.eig(mat)
    si = np.argsort(d)[-1::-1]
    d = d[si]
    U = U[:, si]
    return d, U


def main():
    img = imread('trui.png')

    if not os.path.isfile('d.npy'):
        sum_mat = np.zeros((m*m, m*m), np.int)
        mean_sum_vec = np.zeros((m*m, 1), np.int)

        for i in range(n-m+1):
            for j in range(n-m+1):
                detail = img[i:i+m, j:j+m].reshape((m*m, 1))

                sum_mat += detail * detail.T
                mean_sum_vec += detail

        mean_vec = mean_sum_vec / float(n*n)
        mean_mat = n * mean_vec * mean_vec.T

        S = (sum_mat - mean_mat) / ((m*m)-1)

        d, U = sorted_eig(S)

        np.save('d', d)
        np.save('U', U)
        np.save('mean_vec', mean_vec)
    else:
        d = np.load('d.npy')
        U = np.load('U.npy')
        mean_vec = np.load('mean_vec.npy')

    plt.bar(range(6), d[:6])
    plt.show()

    for k in range(1, 625, 10):
        reconstructed = np.zeros((n, n))

        for i in range(0, 232, 25):
            for j in range(0, 232, 25):
                subimg = compare(img, (i, j), mean_vec, d, U, k)
                reconstructed[i:i+25, j:j+25] = subimg.reshape((25, 25))

        plt.imshow(reconstructed, cmap=cm.Greys_r)
        plt.show()


if __name__ == '__main__':
    main()
