from sklearn.mixture import GMM as SkGMM
from numpy import exp

from semantic.step import Step


class GMM(Step):

    def __init__(self, n_components=5, n_iter=100, n_init=1, **kwargs):
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_init = n_init
        self.kwargs = kwargs
        self.gmm = None

    def fit(self, vectors):
        self.gmm = SkGMM(n_components=self.n_components, n_init=self.n_init,
                         n_iter=self.n_init, **self.kwargs)
        self.gmm.fit(vectors)

    def transform(self, vectors):
        return exp(self.gmm.score(vectors))

    def get_params(self):
        return self.gmm

    def set_params(self, params):
        self.gmm = params
