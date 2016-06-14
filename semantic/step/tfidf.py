from sklearn.feature_extraction.text import TfidfVectorizer
from semantic.step import Step
from semantic.parser import Reader


class TfIdf(Step):

    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, filename):
        article_tokens = Reader(filename)
        article_contents = [' '.join(tokens) for uuid, tokens in article_tokens]
        self.vectorizer.fit(article_contents)

    def transform(self, filename):
        article_tokens = Reader(filename)
        uuids, contents = zip(*[(uuid, ' '.join(tokens)) for uuid, tokens in article_tokens])
        tf_idfs = self.vectorizer.transform(contents)
        return uuids, tf_idfs
