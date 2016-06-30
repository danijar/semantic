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
    definition.corpus = os.path.join(ROOT, definition.corpus)
    return definition


def fitting(vectorizer, corpus):
    print('Fit', type(vectorizer).__name__)
    start = time.time()
    vectorizer.fit(corpus)
    duration = int((time.time() - start) / 60)
    print('Took {} minutes'.format(duration))


def store(vectorizer, corpus, output):
    output = os.path.join(ROOT, output)
    ensure_directory(output)
    name = type(vectorizer).__name__.lower()
    with open(os.path.join(output, '{}.pkl'.format(name)), 'wb') as file_:
        pickle.dump(vectorizer, file_)
    uuids, data = vectorizer.transform(corpus)
    uuids_path = os.path.join(output, 'uuids.npy')
    if os.path.isfile(uuids_path):
        assert (np.load(uuids_path) == uuids).all()
    else:
        np.save(os.path.join(output, 'uuids.npy'), uuids)
    np.save(os.path.join(output, name + '.npy'), data)


def main():
    definition = load_definition()
    for vectorizer in definition.vectorizers:
        fitting(vectorizer, definition.corpus)
        store(vectorizer, definition.corpus, definition.output)


if __name__ == '__main__':
    main()
