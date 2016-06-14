from abc import ABCMeta, abstractmethod


class Step(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError

