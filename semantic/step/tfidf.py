from semantic.step import Step


class Tfidf(Step):

    def __init__(self, min_occurances=3):
        self._min_occurances = min_occurances

    def fit(self, documents):
        pass

    def transform(self, documents):
        pass

    def get_params(self):
        pass

    def set_params(self, params):
        pass
