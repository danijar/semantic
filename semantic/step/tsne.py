import sklearn
from semantic.step import Step


class TSNE(Step):

    def __init__(self, dimensions=2, **kwargs):
        self._dimensions = dimensions
        self._arguments = kwargs

    def fit(self, vectors):
        self.model = sklearn.manifold.TSNE(
            n_components=self.dimensions,
            **self.arguments
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
