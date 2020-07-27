import numpy as np


def create_new_seed():
    return np.random.SeedSequence().entropy
