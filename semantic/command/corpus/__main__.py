import pickle
import os
import definitions
import numpy as np
from semantic.utility import ensure_directory


ROOT = os.path.join(os.path.dirname(__file__), '../../../')


def load_definition():
    schema = os.path.join(os.path.dirname(__file__), 'schema.yaml')
    definition = os.path.join(ROOT, 'definition/corpus.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.fit.filename = os.path.join(ROOT, definition.fit.filename)
    return definition


def main():
    definition = load_definition()
    for vectorizer in definition.vectorizers:
        name = type(vectorizer).__name__.lower()
        output = os.path.join(ROOT, definition.output, name)
        ensure_directory(output)
        print('Fit', name)
        vectorizer.fit(**definition.fit)
        with open(os.path.join(output, 'vectorizer.pkl'), 'wb') as file_:
            pickle.dump(vectorizer, file_)
        uuids, vectors = vectorizer.transform(**definition.transform)
        np.save(os.path.join(output, 'uuids.npy'), uuids)
        np.save(os.path.join(output, 'vectors.npy'), vectors)


if __name__ == '__main__':
    main()
