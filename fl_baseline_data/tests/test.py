# from fl_baseline_data import run

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()
    s = rng.exponential(scale=1, size=100)
    count, bins, ignored = plt.hist(s, 30, density=True)

    plt.show()

    s = rng.multinomial(30000, s / max(s))
    count, bins, ignored = plt.hist(s, 30, density=True)
    # plt.plot(bins,
    #          linewidth=2, color='r')
