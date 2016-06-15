import numpy as np
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from semantic.step import Step


class MultinomialNB(Step):

    def __init__(self, percentile_threshold, bins):
        assert bins > 0
        bin_size = 1 / bins
        self.bins = np.arange(bin_size, 1, bin_size)
        self.lower = percentile_threshold
        self.upper = 100 - percentile_threshold
        scaler = MinMaxScaler()
        discretizer = FunctionTransformer(Discretizer(self.bins))
        self.pipeline = Pipeline(
            [('scaler', scaler), ('discretizer', discretizer)])

    def fit(self, vectors):
        self.lower_clip = np.percentile(vectors, self.lower, axis=0)
        self.upper_clip = np.percentile(vectors, self.upper, axis=0)
        vectors = np.clip(vectors, self.lower_clip, self.upper_clip)
        vectors = self.pipeline.fit_transform(vectors)
        n_docs = vectors.shape[0]
        self.distribution = np.array(
            [np.bincount(v, minlength=len(self.bins)) / n_docs
             for v in vectors.T])

    def transform(self, vectors):
        assert self.distribution is not None
        vectors = np.clip(vectors, self.lower_clip, self.upper_clip)
        probabilities = []
        n_dim = vectors.shape[1]
        vectors = self.pipeline.transform(vectors)
        for bins in vectors:
            pr = np.product(self.distribution[np.arange(n_dim), bins])
            probabilities.append(pr)
        return -np.log(np.maximum(1e-10, np.array(probabilities)))


class MultinomialAVG(Step):

    def __init__(self, percentile_threshold, bins):
        assert bins > 0
        bin_size = 1 / bins
        self.bins = np.arange(bin_size, 1, bin_size)
        self.lower = percentile_threshold
        self.upper = 100 - percentile_threshold
        scaler = MinMaxScaler()
        discretizer = FunctionTransformer(Discretizer(self.bins))
        self.pipeline = Pipeline(
            [('scaler', scaler), ('discretizer', discretizer)])

    def fit(self, vectors):
        self.lower_clip = np.percentile(vectors, self.lower, axis=0)
        self.upper_clip = np.percentile(vectors, self.upper, axis=0)
        vectors = np.clip(vectors, self.lower_clip, self.upper_clip)
        vectors = self.pipeline.fit_transform(vectors)
        n_docs = vectors.shape[0]
        self.distribution = np.array(
            [np.bincount(v, minlength=len(self.bins)) / n_docs
             for v in vectors.T])

    def transform(self, vectors):
        assert self.distribution is not None
        vectors = np.clip(vectors, self.lower_clip, self.upper_clip)
        probabilities = []
        n_dim = vectors.shape[1]
        dp = self.distribution.shape[1] ** (n_dim - 1)
        vectors = self.pipeline.transform(vectors)
        for bins in vectors:
            pr = (self.distribution[np.arange(n_dim), bins]).mean() / dp
            probabilities.append(pr)
        return -np.log(np.maximum(1e-10, np.array(probabilities)))


class MultinomialDEP(Step):

    def __init__(self, percentile_threshold, bins):
        self.lower = percentile_threshold
        self.upper = 100 - percentile_threshold
        scaler = MinMaxScaler()
        discretizer = FunctionTransformer(Discretizer(bins))
        self.pipeline = Pipeline(
            [('scaler', scaler), ('discretizer', discretizer)])

    def fit(self, vectors):
        self.lower_clip = np.percentile(vectors, self.lower, axis=0)
        self.upper_clip = np.percentile(vectors, self.upper, axis=0)
        vectors = np.clip(vectors, self.lower_clip, self.upper_clip)
        self.transformed_vectors = self.pipeline.fit_transform(vectors)

    def transform(self, vectors):
        assert self.transformed_vectors is not None
        vectors = np.clip(vectors, self.lower_clip, self.upper_clip)
        probabilities = []
        vectors = self.pipeline.transform(vectors)
        docs = self.transformed_vectors.shape[0]
        for x in vectors:
            count = np.count_nonzero(
                (self.transformed_vectors == x).all(axis=1))
            pr = count / docs
            probabilities.append(pr)
        return -np.log(np.maximum(1e-10, np.array(probabilities)))


class Discretizer:
    def __init__(self, bins):
        self._bins = bins

    def __call__(self, X):
        return np.digitize(X, bins=self._bins)
