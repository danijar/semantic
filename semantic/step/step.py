from abc import ABCMeta, abstractmethod
import os
import pickle


class Step(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params):
        raise NotImplementedError

    def save(self, filename):
        filepath = os.path.join(self._cache_dir, filename)
        with open(filepath, 'w') as file_:
            pickle.dump(file_, self.get_params())

    def load(self, filename):
        filepath = os.path.join(self._cache_dir, filename)
        with open(filepath) as file_:
            self.set_params(pickle.load(file_))

    @staticmethod
    def _cache_dir(self):
        return os.path.join(os.path.dirname(__file__), '../../cache')
