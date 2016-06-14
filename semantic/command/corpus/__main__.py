import pickle
import os
import definitions
import numpy as np
from semantic.utility import ensure_directory


def main():
    schema = os.path.join(os.path.dirname(__file__), 'schema.yaml')
    root = os.path.join(os.path.dirname(__file__), '../../../')
    definition = os.path.join(root, 'definition/corpus.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.fit.filename = os.path.join(root, definition.fit.filename)
    for vectorizer in definition.vectorizers:
        output = os.path.join(
            root, definition.output, type(vectorizer).__name__.lower())
        ensure_directory(output)
        vectorizer.fit(**definition.fit)
        with open(os.path.join(output, 'vectorizer.pkl'), 'wb') as file_:
            pickel.dump(vectorizer, file_)
        uuids, vectors = vectorizer.transform(**definition.transform)
        np.save(os.path.join(output, 'uuids.npy'), uuids)
        np.save(os.path.join(output, 'vectors.npy'), vectors)


if __name__ == '__main__':
    main()
