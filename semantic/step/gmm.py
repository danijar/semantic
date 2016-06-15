import numpy as np
from sklearn.mixture import GMM as SkGMM
from semantic.step import Step


class GMM(Step):

    def __init__(self, **kwargs):
        self._model = SkGMM(**kwargs)

    def fit(self, vectors):
        self._model.fit(vectors)

    def transform(self, vectors):
        log_density = self._model.score(vectors)
        log_density = np.max(log_density, 1e-10)
        return -log_density
