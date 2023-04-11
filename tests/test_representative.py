import numpy as np
from sklearn.datasets import make_classification
from al_strats.representative import representativeness_score
from tests.common import SEED


def make_data(seed: int):
    """Creates mock unlabeled and train data"""
    data, targets = make_classification(
        n_samples=10, n_features=4, n_redundant=0, n_classes=2, random_state=seed
    )
    return data[targets == 0], data[targets == 1]


def test_representativeness_score():
    data_unlab, data_train = make_data(SEED)

    representativeness = representativeness_score(data_unlab, data_train)
    assert np.argsort(-representativeness).tolist() == [5, 3, 0, 2, 4, 1]
