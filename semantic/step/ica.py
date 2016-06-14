import sklearn
from semantic.step import Step


class ICA(Step):

    def __init__(self, **kwargs):
        self.model = sklearn.decomposition.FastICA(**kwargs)

    def fit(self, vectors):
        self.model.fit(vectors)

    def transform(self, vectors):
        return self.model.transform(vectors)
