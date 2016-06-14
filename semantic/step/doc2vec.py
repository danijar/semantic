import re
import multiprocessing
import gensim
from gensim.models.doc2vec import TaggedDocument
from semantic.step import Step
from semantic.parser import Reader


class Doc2Vec(Step):

    def __init__(self, dimensions=300, iterations=5, **kwargs):
        self._dimensions = dimensions
        self._iterations = iterations
        self._workers = multiprocessing.cpu_count()
        self._keywords = kwargs

    def fit(self, filename):
        self.model = gensim.model.Doc2Vec(
            size=self._dimensions,
            iter=self._iterations,
            workers=self._workers,
            **self._keywords
        )
        self.model.build_vocab(self._read(filename))
        self.model.train(self._read(filename))
        return self.model

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
        return vectors, uuids

    def get_params(self):
        return self.model

    def set_params(self, params):
        self.model = params

    @classmethod
    def _tokenize(cls, text):
        TOKEN_REGEX = re.compile(r'[A-Za-z]+|[0-9]+|[,.!?:]')
        tokens = re.findall(TOKEN_REGEX, text)
        tokens = [x.lower() for x in tokens]
        return tokens

    @classmethod
    def _read(cls, filename):
        reader = Reader(filename)
        for document in reader:
            tokens = cls._tokenize(document.content)
            yield TaggedDocument(tokens, [document.uuid])
