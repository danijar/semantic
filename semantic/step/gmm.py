from sklearn.mixture import GMM as SkGMM
from numpy import exp

from semantic.step import Step


class GMM(Step):

    def __init__(self, **kwargs):
        self.gmm = SkGMM(**kwargs)

    def fit(self, vectors):
        self.gmm.fit(vectors)

    def transform(self, vectors):
        return exp(self.gmm.score(vectors))
