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

    def save(self, filepath):
        with open(filepath, 'wb') as file_:
            pickle.dump(self, file_)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as file_:
            return pickle.load(file_)
