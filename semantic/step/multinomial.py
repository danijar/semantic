from numpy import percentile, clip, digitize, arange, count_nonzero, array
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from semantic.step import Step


class Multinomial(Step):

    def __init__(self, percentile_threshold=1, bins=100):
        self.lower = percentile_threshold
        self.upper = 100 - percentile_threshold
        scaler = MinMaxScaler()
        discretizer = FunctionTransformer(Discretizer(bins))
        self.pipeline = Pipeline([('scaler', scaler),
                                  ('discretizer', discretizer)])

    def fit(self, vectors):
        self.lower_clip = percentile(vectors, self.lower, axis=0)
        self.upper_clip = percentile(vectors, self.upper, axis=0)
        vectors = clip(vectors, self.lower_clip, self.upper_clip)
        self.transformed_vectors = self.pipeline.fit_transform(vectors)

    def transform(self, vectors):
        assert self.transformed_vectors
        vectors = clip(vectors, self.lower_clip, self.upper_clip)
        probabilities = []
        vectors = self.pipeline.transform(vectors)
        docs = self.transformed_vectors.shape[0]
        for x in vectors:
            count = count_nonzero((self.transformed_vectors == x).all(axis=0))
            pr = count / docs
            probabilities.append(pr)
        return array(probabilities)


class Discretizer:
    def __init__(self, bins=100):
        self.bin_size = 1/bins

    def __call__(self, X):
        return digitize(X, bins=arange(self.bin_size, 1, self.bin_size))
