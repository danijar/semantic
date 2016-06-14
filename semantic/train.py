import pickle
import os
import definitions
import numpy as np
from sklearn.cross_validation import KFold
from semantic.utility import ensure_directory


ROOT = os.path.join(os.path.dirname(__file__), '..')


def load_definition():
    schema = os.path.join(ROOT, 'schema/train.yaml')
    definition = os.path.join(ROOT, 'definition/train.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.data = os.path.join(ROOT, definition.data)
    return definition


def load_data(definition):
    with open(os.path.join(ROOT, definition.vectorizer), 'rb') as file_:
        vectorizer = pickle.load(file_)
    _, vectors = vectorizer.transform(definition.data)
    return vectors


def training(definition, distribution, data):
    folds = KFold(
        data.shape[0], n_folds=definition.folds, shuffle=True, random_state=0)
    for train, test in folds:
        train, test = data[train], data[test]
        distribution.fit(train)
        predictions = distribution.transform(test)
        yield -np.log(predictions).sum()


def main():
    definition = load_definition()
    data = load_data(definition)
    for distribution in definition.distributions:
        name = type(distribution).__name__.lower()
        output = os.path.join(ROOT, definition.output, name)
        ensure_directory(output)
        costs = np.array(list(training(definition, distribution, data)))
        message = 'Fit {} cost mean {} variance {}'
        print(message.format(costs.mean(), costs.std(), name))
        distribution.fit(data)
        with open(os.path.join(output, 'distribution.pkl'), 'wb') as file_:
            pickle.dump(distribution, file_)


if __name__ == '__main__':
    main()
