from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # noqa


def main():
    data = pd.read_csv('biometrie2014.csv')
    print(data)
    print('\n\nstd\n')
    print(data.std())
    print('\n\nmean\n')
    print(data.mean())
    for name, group in data.groupby('Man/Vrouw'):
        print(name)
        print(group)
        print('std', group.std())
        print('mean', group.mean())

if __name__ == '__main__':
    main()
