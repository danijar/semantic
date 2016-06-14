import pickle
import os
import definitions
from semantic.utility import ensure_directory


def load_definition(root):
    schema = os.path.join(os.path.dirname(__file__), 'schema.yaml')
    definition = os.path.join(root, 'definition/train.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.data = os.path.join(root, definition.data)
    return definition


def train(distribution, vectors):
    distribution.fit(vectors)


def test(distribution, vectors):
    pass


def main():
    root = os.path.join(os.path.dirname(__file__), '../../..')
    definition = load_definition(root)
    with open(os.path.join(root, definition.vectorizer), 'rb') as file_:
        vectorizer = pickle.load(file_)
    _, vectors = vectorizer.transform(definition.data)
    # TODO: Split dataset.
    for distribution in definition.distributions:
        output = os.path.join(
            root, definition.output, type(distribution).__name__.lower())
        ensure_directory(output)
        train(distribution, train_data)
        test(distribution, test_data)
        with open(os.path.join(output, 'distribution.pkl'), 'wb') as file_:
            pickle.dump(distribution, file_)


if __name__ == '__main__':
    main()
