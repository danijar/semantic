import pickle
import os
import matplotlib.pyplot as plt
import definitions


def load_definition(root):
    schema = os.path.join(os.path.dirname(__file__), 'schema.yaml')
    definition = os.path.join(root, 'definition/visualize.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.data = os.path.join(root, definition.data)
    return definition


def plot(vectors):
    fig, ax = plt.subplots()
    ax.scatter(vectors[:, 0], vectors[:, 1])
    plt.show()


def main():
    root = os.path.join(os.path.dirname(__file__), '../../..')
    definition = load_definition(root)
    with open(os.path.join(root, definition.vectorizer), 'rb') as file_:
        vectorizer = pickle.load(file_)
    _, vectors = vectorizer.transform(definition.data)
    plot(vectors)


if __name__ == '__main__':
    main()
