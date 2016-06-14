from abc import ABCMeta, abstractmethod


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
        pass

    def load(self, filename):
        pass
