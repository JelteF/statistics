from scipy.misc import comb


def exp(p, n):
    total = 0.0
    for k in range(n+1):
        total += comb(n, k, exact=False) * p**k * (1-p) ** (n-k)

    return total


def main():
    for p in [0.3, 0.75, 0.8, 1.0, 0.0, 0.5]:
        for n in range(1, 20):
            print('Checking n=%d, p=%f' % (n, p))
            print('Result: %f' % (exp(p, n)))

if __name__ == '__main__':
    main()
