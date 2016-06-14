import multiprocessing
import gensim
from gensim.models.doc2vec import TaggedDocument
from semantic.parser import Reader
from semantic.step import Step


class Doc2Vec(Step):

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._model = gensim.models.Doc2Vec(
            workers=multiprocessing.cpu_count(),
            **kwargs)

    def fit(self, filename):
        self._model.build_vocab(self._read(filename))
        self._model.train(self._read(filename))
        return self._model

    def transform(self, filename):
        assert self.model
        vectors = []
        uuids = []
        for tokens, uuid in self._read(filename):
            uuid = uuid[0]
            if uuid in self.model.docvecs:
                vector = self.model.docvecs[uuid]
            else:
                vector = self.model.infer_vector(tokens)
            vectors.append(vector)
            uuids.append(uuid)
        return uuids, vectors

    def get_params(self):
        return self._model

    def set_params(self, params):
        self._model = params

    @classmethod
    def _read(cls, filename):
        for uuid, tokens in Reader(filename):
            yield TaggedDocument(tokens, [uuid])
