def ibm_rng(x1, a=65539, c=0, m=2**31):
    x = x1
    while True:
        x = (a * x + c) % m
        yield x / (m-1)


def main():
    rng = ibm_rng(1, 65539, 0, 2**31)

    while True:
        x = next(rng)
        print(x)


if __name__ == '__main__':
    main()
