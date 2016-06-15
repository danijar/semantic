import numpy as np
import gensim
from sklearn.decomposition import PCA
from gensim.models.doc2vec import TaggedDocument
from semantic.parser import Reader
from semantic.step import Step


class Doc2Vec(Step):

    def __init__(self, pca=None, **kwargs):
        self._model = gensim.models.Doc2Vec(**kwargs)
        self._pca = PCA(**(pca or {}))

    def fit(self, filename):
        self._model.build_vocab(self._read(filename))
        self._model.train(self._read(filename))
        _, vectors = self._transform(filename)
        self._pca.fit(vectors)

    def transform(self, filename):
        uuids, vectors = self._transform(filename)
        vectors = self._pca.transform(vectors)
        return uuids, vectors

    def _transform(self, filename):
        vectors = []
        uuids = []
        for uuid, tokens in Reader(filename):
            if uuid in self._model.docvecs:
                vector = self._model.docvecs[uuid]
            else:
                vector = self._model.infer_vector(tokens)
            vectors.append(vector)
            uuids.append(uuid)
        vectors = np.array(vectors)
        return uuids, vectors

    @classmethod
    def _read(cls, filename):
        for uuid, tokens in Reader(filename):
            yield TaggedDocument(tokens, [uuid])
