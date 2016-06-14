from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from semantic.step import Step
from semantic.parser import Reader


class TfIdf(Step):

    def __init__(self, svd=None, **kwargs):
        self._vectorizer = TfidfVectorizer(**kwargs)
        self._svd = TruncatedSVD(**(svd or {}))

    def fit(self, filename):
        articles = Reader(filename)
        contents = [' '.join(x) for _, x in articles]
        vectors = self._vectorizer.fit_transform(contents)
        self._svd.fit(vectors)

    def transform(self, filename):
        articles = Reader(filename)
        uuids, contents = zip(*[(x, ' '.join(y)) for x, y in articles])
        vectors = self._vectorizer.transform(contents)
        vectors = self._svd.transform(vectors)
        return uuids, vectors
