import numpy as np
import gensim
from sklearn.decomposition import FastICA
from gensim.models.doc2vec import TaggedDocument
from semantic.parser import Reader
from semantic.step import Step


class Doc2Vec(Step):

    def __init__(self, ica=None, **kwargs):
        self._model = gensim.models.Doc2Vec(**kwargs)
        self._ica = FastICA(**(ica or {}))

    def fit(self, filename):
        self._model.build_vocab(self._read(filename))
        # self._model.scale_vocab()
        # self._model.finalize_vocab()
        self._model.train(self._read(filename))
        print('Doc2Vec log accuracy', self._model.log_accuracy())
        _, vectors = self.transform(filename)
        self._ica.fit(vectors)

    def transform(self, filename):
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
        vectors = self._ica.transform(vectors)
        return uuids, vectors

    @classmethod
    def _read(cls, filename):
        for uuid, tokens in Reader(filename):
            yield TaggedDocument(tokens, [uuid])
