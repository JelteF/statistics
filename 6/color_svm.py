from sklearn import preprocessing
from sklearn import svm
import numpy as np
import numpy.random as rd


def main(seed=None):
    colors = ['black', 'blue', 'brown', 'gray',
              'green', 'orange', 'pink', 'red',
              'violet', 'white', 'yellow']

    # does a line of text contains a color name?
    def containsColor(line):
        for c in colors:
            if line.startswith('#' + c):
                return colors.index(c)
        return None

    # read the file and store spectra in matrix D (rows are the spectra)
    # and the classes in vector y
    fp = open("data/natural400_700_5.asc", encoding='utf-8')
    X = np.zeros((0, 61))
    y = np.array([])

    # Iterator over the list of lines in pairs of two
    line_iterator = iter(fp)

    for comment, vec in zip(line_iterator, line_iterator):
        ind = containsColor(comment)
        if ind is not None:
            d = np.fromstring(vec, dtype=int, sep=" ")
            X = np.append(X, np.array([d]), axis=0)
            y = np.append(y, ind)

    print(X)
    # Split data into a matrix X containing all RVs, and y, containing all
    # classes.

    # Scale data.
    X_scaled = preprocessing.scale(X)

    # Choose learn and test data.
    if seed is not None:
        rd.seed(seed)
    ind = np.arange(150)  # indices into the dataset
    ind = rd.permutation(ind)  # random permutation
    L = ind[0:90]  # learning set indices
    T = ind[90:]  # test set indices

    # Learning set.
    X_learn = X_scaled[L, :]
    y_learn = y[L]

    # Create SVM.
    clf = svm.SVC()

    # Fit data.
    clf.fit(X_learn, y_learn)

    # Test set.
    X_test = X_scaled[T, :]
    y_test = y[T]

    # Test all data.
    pred_clss = clf.predict(X_test)

    # Create confusion matrix.
    cm = np.zeros((3, 3))
    for pred_cls, cls in zip(pred_clss, y_test):
        cm[cls, pred_cls] += 1

    print(cm)


def cnvt(s):
    tab = {'Iris-setosa': 0.0, 'Iris-versicolor': 1.0, 'Iris-virginica': 2.0}
    s = s.decode()
    if s in tab:
        return tab[s]
    else:
        return -1.0


if __name__ == '__main__':
    main()
