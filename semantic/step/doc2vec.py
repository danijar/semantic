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
        assert self._model
        vectors = []
        uuids = []
        for tokens, uuid in self._read(filename):
            uuid = uuid[0]
            if uuid in self._model.docvecs:
                vector = self._model.docvecs[uuid]
            else:
                vector = self._model.infer_vector(tokens)
            vectors.append(vector)
            uuids.append(uuid)
        return uuids, vectors

    @classmethod
    def _read(cls, filename):
        for uuid, tokens in Reader(filename):
            yield TaggedDocument(tokens, [uuid])
