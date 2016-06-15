import itertools
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


def load_sources(definition):
    for filepath in definition.vectorizers:
        print('Load vectorizer', filepath)
        with open(os.path.join(ROOT, filepath), 'rb') as file_:
            vectorizer = pickle.load(file_)
        print('Transform user corpus')
        _, data = vectorizer.transform(definition.data)
        name = '.'.join(os.path.basename(filepath).split('.')[:-1])
        yield name, data


def training(distribution, data, folds):
    folds = KFold(data.shape[0], n_folds=folds, shuffle=True, random_state=0)
    for train, test in folds:
        train, test = data[train], data[test]
        distribution.fit(train)
        yield distribution.transform(test).mean()


def store_distribution(distribution, vectorizer, name, output):
        output = os.path.join(ROOT, output)
        ensure_directory(output)
        filename = '{}-{}.pkl'.format(vectorizer.lower(), name.lower())
        with open(os.path.join(output, filename), 'wb') as file_:
            pickle.dump(distribution, file_)


def main():
    definition = load_definition()
    sources = load_sources(definition)
    combinations = itertools.product(sources, definition.distributions)
    for (vectorizer, data), distribution in combinations:
        name = type(distribution).__name__
        print('Fit {} with {}'.format(vectorizer, name))
        cost = np.array(list(training(distribution, data, definition.folds)))
        message = 'Cost on test mean {:6.2f} std {:6.4f}'
        print(message.format(cost.mean(), cost.std()))
        distribution.fit(data)
        store_distribution(distribution, vectorizer, name, definition.output)


if __name__ == '__main__':
    main()
