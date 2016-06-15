from sklearn.preprocessing import normalize

from semantic.step import Step


class CoSim(Step):
    def __init__(self, n_nearest):
        assert n_nearest > 0
        self.n_nearest = n_nearest

    def fit(self, vectors):
        self.vectors = normalize(vectors, norm='l2')

    def transform(self, vectors):
        assert vectors.shape[1] == self.vectors.shape[1]
        vectors = normalize(vectors, norm='l2')
        sims = vectors.dot(self.vectors.T)
        sims.sort(axis=1)
        sims = sims[:, -self.n_nearest:]
        return sims.mean(axis=1)
