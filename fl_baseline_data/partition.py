import numpy as np
from scipy import stats


class Partition:
    def __init__(self, num_elements, num_splits, min_value=0,
                 distribution='exponential', distribution_params=None,
                 seed=None):
        rng = np.random.default_rng(seed)

        weight_per_split = getattr(rng, distribution)(size=num_splits, **(distribution_params or {}))

        # normalize into probabilities
        weight_per_split = weight_per_split - np.min(weight_per_split)
        pvals = weight_per_split / np.linalg.norm(weight_per_split, ord=1)

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
    def lognormal(cls, num_elements, num_splits, mean, sigma, min_value=0, seed=None):
        return cls(num_elements, num_splits, min_value=min_value,
                   distribution='lognormal', distribution_params={'mean': mean, 'sigma': sigma},
                   seed=seed)


