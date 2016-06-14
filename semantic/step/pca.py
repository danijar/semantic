import sklearn
from semantic.step import Step


class PCA(Step):

    def __init__(self, dimensions=50, kernel='rbf', **kwargs):
        self._dimensions = dimensions
        self._kernel = kernel
        self._arguments = kwargs

    def fit(self, vectors):
        self.model = sklearn.decomposition.KernelPCA(
            n_components=self._dimensions,
            kernel=self._kernel,
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
