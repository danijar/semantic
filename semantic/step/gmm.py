from sklearn.mixture import GMM as SkGMM
import numpy as np

from semantic.step import Step


class GMM(Step):

    def __init__(self, **kwargs):
        self.gmm = SkGMM(**kwargs)

    def fit(self, vectors):
        self.gmm.fit(vectors)

    def transform(self, vectors):
        return np.max(self.gmm.predict_proba(vectors), axis=1)
