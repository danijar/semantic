import os
import definitions
import numpy as np
from semantic.utility import ensure_directory


def main():
    schema = os.path.join(os.path.dirname(__file__), 'schema.yaml')
    definition = os.path.join(
        os.path.dirname(__file__), '../../../definition/corpus.yaml')
    definition = definitions.Parser(schema)(definition)
    output = os.path.join(
        os.path.dirname(__file__), '../../..', definition.output)
    ensure_directory(output)
    for vectorizer in definition.vectorizers:
        directory = os.path.join(output, type(vectorizer).__name__.lower())
        vectorizer.fit(**definition.fit)
        vectorizer.save(os.path.join(directory, 'params.pkl'))
        uuids, vectors = vectorizer.transform(**definition.transform)
        np.save(os.path.join(directory, 'uuids.npy'), uuids)
        np.save(os.path.join(directory, 'vectors.npy'), vectors)


if __name__ == '__main__':
    main()
