import pickle
import os
import time
import definitions
import numpy as np
from semantic.utility import ensure_directory


ROOT = os.path.join(os.path.dirname(__file__), '..')


def load_definition():
    schema = os.path.join(ROOT, 'schema/embed.yaml')
    definition = os.path.join(ROOT, 'definition/embed.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.fit.filename = os.path.join(ROOT, definition.fit.filename)
    return definition


def fitting(vectorizer, **kwargs):
    print('Fit', type(vectorizer).__name__)
    start = time.time()
    vectorizer.fit(**kwargs)
    duration = int((time.time() - start) / 60)
    print('Took {} minutes'.format(duration))


def store(vectorizer, output, **kwargs):
    ensure_directory(output)
    with open(os.path.join(output, 'vectorizer.pkl'), 'wb') as file_:
        pickle.dump(vectorizer, file_)
    uuids, vectors = vectorizer.transform(**kwargs)
    np.save(os.path.join(output, 'uuids.npy'), uuids)
    np.save(os.path.join(output, 'vectors.npy'), vectors)


def main():
    definition = load_definition()
    for vectorizer in definition.vectorizers:
        name = type(vectorizer).__name__.lower()
        fitting(vectorizer, **definition.fit)
        output = os.path.join(ROOT, definition.output, name)
        store(vectorizer, output, **definition.transform)


if __name__ == '__main__':
    main()
