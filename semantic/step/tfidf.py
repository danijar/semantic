from sklearn.feature_extraction.text import TfidfVectorizer
from semantic.step import Step
from semantic.parser import Reader


class TfIdf(Step):

    def __init__(self, **kwargs):
        self._vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, filename):
        article_tokens = Reader(filename)
        article_contents = [' '.join(tokens) for uuid, tokens in article_tokens]
        print(type(article_contents), len(article_contents))
        self._vectorizer.fit(article_contents)

    def transform(self, filename):
        article_tokens = Reader(filename)
        uuids, contents = zip(*[(uuid, ' '.join(tokens)) for uuid, tokens in article_tokens])
        print(type(contents), len(contents))
        tf_idfs = self._vectorizer.transform(contents)
        return uuids, tf_idfs

    def get_params(self):
        return self._vectorizer.get_params()

    def set_params(self, params):
        self._vectorizer.set_params(**params)
