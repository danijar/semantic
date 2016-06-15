from sklearn.mixture import GMM as SkGMM
from semantic.step import Step


class GMM(Step):

    def __init__(self, **kwargs):
        self._model = SkGMM(**kwargs)

    def fit(self, vectors):
        self._model.fit(vectors)
        message = '  GMM aic score {} (lower means better fit)'
        print(message.format(self._model.aic(vectors)))

    def transform(self, vectors):
        return self._model.score(vectors)
