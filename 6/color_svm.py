from sklearn import preprocessing
from sklearn import svm
from sklearn import grid_search
from sklearn import cross_validation
from itertools import combinations_with_replacement
from functools import partial
import numpy as np


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

    # Scale data.
    X_scaled = preprocessing.scale(X)

    # Choose learn and test data.
    X_learn, X_test, y_learn, y_test = \
        cross_validation.train_test_split(X_scaled, y, test_size=0.5,
                                          random_state=seed)

    # Create SVM.
    clf = svm.SVC()

    # Brute-force parameters
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    params = {'C': C_range, 'gamma': gamma_range}
    clf_p = grid_search.GridSearchCV(clf, params, cv=3)

    # Fit data.
    clf.fit(X_learn, y_learn)
    clf_p.fit(X_learn, y_learn)

    # Test all data.
    pred_clss = clf.predict(X_test)
    pred_clss_p = clf_p.predict(X_test)

    # Create confusion matrix.
    # cm = np.zeros((len(colors), len(colors)))
    # cm_p = np.zeros((len(colors), len(colors)))
    good = 0
    good_p = 0
    for pred_cls, pred_cls_p, cls in zip(pred_clss, pred_clss_p, y_test):
        good += pred_cls == cls
        good_p += pred_cls_p == cls
        # cm[cls, pred_cls] += 1
        # cm_p[cls, pred_cls_p] += 1

    print('%d/%d (%f%%) good' % (good, len(y_test), good / len(y_test) * 100))
    # print(cm)
    print('%d/%d (%f%%) good (p)' % (good_p, len(y_test),
                                     good_p / len(y_test) * 100))

    print(clf_p.best_estimator_, clf_p.best_score_, clf_p.best_params_)
    # print(cm_p)

    return good_p / len(y_test)


def cnvt(s):
    tab = {'Iris-setosa': 0.0, 'Iris-versicolor': 1.0, 'Iris-virginica': 2.0}
    s = s.decode()
    if s in tab:
        return tab[s]
    else:
        return -1.0


if __name__ == '__main__':
    seed = 1
    main(seed=seed)

    good = 0
    n = 50
    for i in range(n):
        print('Test %d' % (i+1))
        good += main()

    print('Total: %f%%' % (good / n * 100))
