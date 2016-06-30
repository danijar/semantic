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
        contents = [x for _, x in Reader(filename)]
        self._dictionary = Dictionary(contents)
        corpus = [self._dictionary.doc2bow(text) for text in contents]
        self._model = LdaModel(corpus, num_topics=self._n_topics)

    def transform(self, filename):
        uuids, vectors = self._transform(filename)
        return uuids, vectors

    def _transform(self, filename):
        vectors = []
        uuids = []
        for uuid, tokens in Reader(filename):
            bow = self._dictionary.doc2bow(tokens)
            vectors.append(self._model[bow])
            uuids.append(uuid)
        return uuids, np.array(vectors)

    @classmethod
    def _read(cls, filename):
        for uuid, tokens in Reader(filename):
            yield ' '.join(tokens)
