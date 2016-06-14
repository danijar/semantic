from sklearn.feature_extraction.text import TfidfVectorizer
from semantic.step import Step
from semantic.parser import Reader


class TfIdf(Step):

    def __init__(self, **kwargs):
        self._vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, filename):
        articles = Reader(filename)
        articles = [''.join(x.tokens) for x in articles]
        self._vectorizer.fit(articles)

    def transform(self, filename):
        articles = Reader(filename)
        uuids, contents = zip(*[(a.uuid, ''.join(a.tokens)) for a in articles])
        tf_idfs = self._vectorizer.transform(contents)
        return uuids, tf_idfs

    def get_params(self):
        return self._vectorizer.get_params()

    def set_params(self, params):
        self._vectorizer.set_params(**params)
