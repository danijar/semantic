import sklearn
from semantic.step import Step


class TSNE(Step):

    def __init__(self, dimensions=2, **kwargs):
        self.model = sklearn.manifold.TSNE(**kwargs)

    def fit(self, vectors):
        self.model.fit(vectors)
        return self.model

    def transform(self, vectors):
        return self.model.transform(vectors)
