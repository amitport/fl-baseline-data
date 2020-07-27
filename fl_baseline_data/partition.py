import numpy as np
from scipy import stats
from scipy.special import softmax
from collections import namedtuple


# implemented 3 ways to map random weights (concentrations?) into pvals

def l1(weights, rng):
    weights -= np.min(weights)
    return weights / np.sum(weights)


def dirichlet(weights, rng):
    # weights -= np.min(weights)
    return rng.dirichlet(weights)


pvals_from_weights = {
    'l1': l1,
    'dirichlet': dirichlet,
    'softmax': lambda weights, rng: softmax(weights),
}


def pvals_from_weights_2(weights, rng):
    return rng.dirichlet(weights)


def pvals_from_weights_3(weights):
    return softmax(weights)


AllEqualPartition = namedtuple('AllEqualPartition', ['parts', 'apply_to_array'])


class Partition:
    def __init__(self, num_elements, num_splits,
                 distribution, pvals_fn, min_value=0, distribution_params=None, seed=None):
        rng = np.random.default_rng(seed)

        weight_per_split = getattr(rng, distribution)(size=num_splits, **(distribution_params or {}))

        # normalize into probabilities
        pvals = pvals_from_weights[pvals_fn](weight_per_split, rng)

        if num_elements < min_value * num_splits:
            raise Exception(f'''\
{num_splits} with minimum {min_value} per split requires {min_value * num_splits} elements \
- but only {num_elements} were given''')

        parts = rng.multinomial(num_elements - min_value * num_splits, pvals) + min_value
        self.parts = parts

    def describe(self):
        return stats.describe(self.parts)

    def apply_to_array(self, arr):
        sums = np.cumsum(self.parts)
        return np.split(arr[:sums[-1]], sums[:-1])

    @classmethod
    def all_equal(cls, num_elements, num_splits):
        size_per_split = num_elements // num_splits
        parts = np.repeat(size_per_split, num_splits)
        return AllEqualPartition(parts=parts, apply_to_array=lambda arr: np.split(arr, num_splits))

    @classmethod
    def uniform_like(cls, num_elements, num_splits, min_value=0, seed=None):
        return cls(num_elements, num_splits, min_value=min_value,
                   distribution='uniform',
                   pvals_fn='l1',
                   seed=seed)

    @classmethod
    def normal_like(cls, num_elements, num_splits, mean, sigma, min_value=0, seed=None):
        return cls(num_elements, num_splits, min_value=min_value,
                   distribution='normal', distribution_params={'loc': mean, 'scale': sigma},
                   pvals_fn='l1',
                   seed=seed)

    @classmethod
    def lognormal_like(cls, num_elements, num_splits, mean, sigma, min_value=0, seed=None):
        return cls(num_elements, num_splits, min_value=min_value,
                   distribution='normal', distribution_params={'loc': mean, 'scale': sigma},
                   pvals_fn='softmax',
                   seed=seed)

    @classmethod
    def pareto_like(cls, num_elements, num_splits, shape, min_value=0, seed=None):
        return cls(num_elements, num_splits, min_value=min_value,
                   distribution='exponential', distribution_params={'scale': 1 / shape},
                   pvals_fn='softmax',
                   seed=seed)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_splits = 10000
    min_value = 60

    p = Partition.all_equal(1000000, num_splits)
    plt.hist(p.parts, bins=1000)
    plt.show()

    p = Partition.uniform_like(1000000, num_splits, min_value=min_value)
    plt.hist(p.parts, bins=1000)
    plt.show()

    p = Partition.normal_like(1000000, num_splits, min_value=min_value, mean=20, sigma=1)
    plt.hist(p.parts, bins=1000)
    plt.show()

    p = Partition.lognormal_like(1000000, num_splits, min_value=min_value, mean=20, sigma=1)
    plt.hist(p.parts, bins=1000)
    plt.show()

    p = Partition.pareto_like(1000000, num_splits, min_value=min_value, shape=1.9)
    plt.hist(p.parts, bins=1000)
    plt.show()
