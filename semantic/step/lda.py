import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from semantic.step import Step
from semantic.parser import Reader


class LDA(Step):

    def __init__(self, num_topics):
        self._model = None
        self._dictionary = None
        self._n_topics = num_topics

    def fit(self, filename):
        contents = [' '.join(x) for _, x in Reader(filename)]
        self._dictionary = Dictionary(contents)
        self._model = LdaModel(self._dictionary, num_topics=self._n_topics)

    def transform(self, filename):
        vectors = self._transform(filename)
        return vectors

    def _transform(self, filename):
        vectors = []
        for _, tokens in Reader(filename):
            bow = self._dictionary[tokens]
            vectors.append(self._model[bow])
        return np.array(vectors)

    @classmethod
    def _read(cls, filename):
        for uuid, tokens in Reader(filename):
            yield ' '.join(tokens)
