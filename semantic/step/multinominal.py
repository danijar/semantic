from numpy import percentile, clip, digitize, arange, count_nonzero, array
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from semantic.step import Step


class Multinominal(Step):

    def __init__(self, percentile_threshold=1, bins=100):
        self.lower = percentile_threshold
        self.upper = 100 - percentile_threshold
        self.bins = bins
        self.lower_clip = None
        self.upper_clip = None
        self.pipeline = None
        self.transformed_vectors = None

    def fit(self, vectors, percentile_threshold=1):
        self.lower_clip = percentile(vectors, self.lower, axis=0)
        self.upper_clip = percentile(vectors, self.upper, axis=0)
        vectors = clip(vectors, self.lower_clip, self.upper_clip)
        scaler = MinMaxScaler()
        discretizer = FunctionTransformer(Discretizer(self.bins))
        self.pipeline = Pipeline([('scaler', scaler),
                                  ('discretizer', discretizer)])
        self.transformed_vectors = self.pipeline.fit_transform(vectors)

    def transform(self, vectors):
        assert self.pipeline is not None
        probabilities = []
        vectors = self.pipeline.transform(vectors)
        docs = self.transformed_vectors.shape[0]
        for x in vectors:
            count = count_nonzero((self.transformed_vectors == x).all(axis=0))
            pr = count / docs
            probabilities.append(pr)
        return array(probabilities)

    def get_params(self):
        return self.pipeline, self.transformed_vectors

    def set_params(self, params):
        self.pipeline, self.transformed_vectors = params


class Discretizer:
    def __init__(self, bins=100):
        self.bin_size = 1/bins

    def __call__(self, X):
        return digitize(X, bins=arange(self.bin_size, 1, self.bin_size))
