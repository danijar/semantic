from sklearn.feature_extraction.text import TfidfVectorizer

from semantic.step import Step


class TfIdf(Step):

    def __init__(self, min_occurrences=3, **params):
        # TODO: min_occurrences?
        # self._min_occurrences = min_occurrences
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.15,
            min_df=0.05,
            **params
        )

    def fit(self, articles):
        articles = [a.content for a in articles]
        self.vectorizer.fit(articles)

    def transform(self, articles):
        uuids, articles = zip(*[(a.uuid, a.content) for a in articles])
        tf_idfs = self.vectorizer.transform(articles['content'])
        return uuids, tf_idfs

    def get_params(self):
        return self.vectorizer.get_params()

    def set_params(self, params):
        self.vectorizer.set_params(**params)
