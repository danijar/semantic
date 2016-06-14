import sklearn
from semantic.step import Step


class ICA(Step):

    def __init__(self, dimensions=50, **kwargs):
        self._dimensions = dimensions
        self._arguments = kwargs

    def fit(self, vectors):
        self.model = sklearn.decomposition.FastICA(
            n_components=self._dimensions,
            **self._arguments
        )
        self.model.fit(vectors)
        return self.model

    def transform(self, vectors):
        assert self.model
        return self.model.transform(vectors)

    def get_params(self):
        return self.model

    def set_params(self, params):
        self.model = params
