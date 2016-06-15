import numpy as np
from semantic.step import Step
from sklearn.preprocessing import normalize


class NearestNeighbors(Step):

    def __init__(self, neighbors):
        self._neighbors = neighbors

    def fit(self, data):
        self._train = normalize(data)

    def transform(self, data):
        assert data.shape[1] == self._train.shape[1]
        data = normalize(data)
        distances = self._distance(data, self._train)
        distances.sort(axis=1)
        distances = distances[:, -self._neighbors:]
        return np.square(distances.mean(axis=1))

    @staticmethod
    def _distance(data, reference):
        return data.dot(reference.T)
