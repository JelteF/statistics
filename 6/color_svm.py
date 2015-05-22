from sklearn import preprocessing
from sklearn import svm
from sklearn import grid_search
from itertools import combinations_with_replacement
from functools import partial
import time
import numpy as np
import numpy.random as rd


def main(seed=None, with_combinations=False):
    colors = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink',
              'red', 'violet', 'white', 'yellow']

    # does a line of text contains a color name?
    def containsColor(line):
        for c in colors:
            if line.startswith('#' + c):
                return colors.index(c)
        return None

    # read the file and store spectra in matrix D (rows are the spectra)
    # and the classes in vector y
    fp = open('data/natural400_700_5.asc', encoding='utf-8')
    X = np.zeros((0, 61))
    y = np.array([])

    # Iterator over the lines in pairs of two
    # http://stackoverflow.com/a/2990873/2570866
    line_iterator = iter(fp)

    # Read data, using two lines per datapoint: the first for the class vector
    # y, and the second for matrix X.
    for comment, vec in zip(line_iterator, line_iterator):
        ind = containsColor(comment)
        if ind is not None:
            d = np.fromstring(vec, dtype=int, sep=' ')
            X = np.append(X, np.array([d]), axis=0)
            y = np.append(y, ind)

    if with_combinations:
        n = X.shape[0]
        for i, j in combinations_with_replacement(range(61), 2):
            a = X[:, i] * X[:, j]
            X = np.hstack((X, a.reshape((n, 1))))

    # Scale data.
    X_scaled = preprocessing.scale(X)
    # print(X_scaled)

    # Choose learn and test data.
    if seed is not None:
        rd.seed(seed)
    ind = np.arange(X_scaled.shape[0])  # indices into the dataset
    ind = rd.permutation(ind)  # random permutation
    L = ind[0:90]  # learning set indices
    T = ind[90:]  # test set indices

    # Learning set.
    X_learn = X_scaled[L, :]
    y_learn = y[L]

    # Create SVM.
    clf = svm.SVC()

    # Brute-force parameters
    params = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
               'kernel': ['rbf']}, ]
    clf_p = grid_search.GridSearchCV(clf, params)

    # Fit data.
    clf.fit(X_learn, y_learn)
    clf_p.fit(X_learn, y_learn)

    # Test set.
    X_test = X_scaled[T, :]
    y_test = y[T]

    # Test all data.
    pred_clss = clf.predict(X_test)
    pred_clss_p = clf_p.predict(X_test)

    # Create confusion matrix.
    cm = np.zeros((len(colors), len(colors)))
    cm_p = np.zeros((len(colors), len(colors)))
    good = 0
    good_p = 0
    for pred_cls, pred_cls_p, cls in zip(pred_clss, pred_clss_p, y_test):
        good += pred_cls == cls
        good_p += pred_cls_p == cls
        cm[cls, pred_cls] += 1
        cm_p[cls, pred_cls_p] += 1

    print('%d/%d (%f%%) good' % (good, len(y_test), good / len(y_test) * 100))
    print(cm)
    print('%d/%d (%f%%) good (p)' % (good_p, len(y_test),
                                     good_p / len(y_test) * 100))
    print(cm_p)


def cnvt(s):
    tab = {'Iris-setosa': 0.0, 'Iris-versicolor': 1.0, 'Iris-virginica': 2.0}
    s = s.decode()
    if s in tab:
        return tab[s]
    else:
        return -1.0


if __name__ == '__main__':
    p_main = partial(main, seed=int(time.time()))
    print(p_main)
    p_main()
    p_main(with_combinations=True)
