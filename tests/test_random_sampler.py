import numpy as np
from al_strats.sampler import RandomSampler
from tests.common import SEED


def test_random_sampler():
    sampler = RandomSampler(SEED)
    sample_idx = sampler(5, 2)
    assert all(sample_idx == np.array([0, 3]))


def test_random_sampler_dataset_size_overflow():
    sampler = RandomSampler(SEED)
    sample_idx = sampler(5, 6)
    print(sample_idx)
    assert all(sample_idx == np.array([4, 1, 0, 3, 2]))
