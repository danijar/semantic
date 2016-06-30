from sklearn.mixture import DPGMM
from semantic.step import Step


class GMM(Step):

    def __init__(self, **kwargs):
        self._model = DPGMM(**kwargs)

    def fit(self, vectors):
        self._model.fit(vectors)

    def transform(self, vectors):
        return self._model.score(vectors)
