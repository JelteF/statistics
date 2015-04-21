from random import sample
from numpy import std
from math import sqrt


TEST_CASE_N = 100
TEST_DATA_N = 50
C = 2.009
ALPHA = 0.05


def main():
    with open('tijden-medium.log', 'r') as f:
        data = []

        for line in f:
            data.append(float(line.rstrip('\n')))

        confidence(data)


def confidence(data):
    total_mu = sum(data) / len(data)

    hits = 0

    for i in range(TEST_CASE_N):
        if test_case(data, total_mu):
            hits += 1

    print('Estimated chance: %f' % (1-ALPHA))
    print('Hits: %d, misses: %d, calculated chance: %f' % (hits,
                                                           TEST_CASE_N-hits,
                                                           hits/TEST_CASE_N))


def test_case(data, mu):
    test_data = sample(data, TEST_DATA_N)

    x_avg = sum(test_data) / TEST_DATA_N
    s = std(test_data)

    t = (x_avg - mu) / (s / sqrt(TEST_DATA_N))

    hit = t >= -C and t <= C

    print('$%f \leq %f \leq %f$ & %s \\\\' % (-C, t, C,
                                              'binnen interval' if hit
                                              else 'buiten interval'))

    return hit


if __name__ == '__main__':
    main()
