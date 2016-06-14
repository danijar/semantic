import pickle
import os
import definitions
import numpy as np
import matplotlib.pyplot as plt
from semantic.utility import ensure_directory


ROOT = os.path.join(os.path.dirname(__file__), '..')


def load_definition():
    schema = os.path.join(ROOT, 'schema/visualize.yaml')
    definition = os.path.join(ROOT, 'definition/visualize.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.output = os.path.join(ROOT, definition.output)
    for source in definition.source:
        source.filename = os.path.join(ROOT, source.filename)
    return definition


def load_data(definition):
    with open(os.path.join(ROOT, definition.vectorizer), 'rb') as file_:
        vectorizer = pickle.load(file_)
    vectors, colors = [], []
    for source in definition.sources:
        _, vectors = vectorizer.transform(source.filename)
        vectors.append(vectors)
        colors.append(source.color)
    return vectors, colors


def plot_vectors(data, colors, labels, output):
    fig, ax = plt.subplots(figsize=(12, 8))
    for points, label, color in zip(data, labels, colors):
        ax.scatter(points, label=label, c=color)
    ax.legend(loc='upper right')
    ensure_directory(output)
    filename = os.path.join(output, 'figure.png')
    fig.savefig(filename, dpi=300)


def main():
    definition = load_definition()
    data, colors = load_data(definition)
    lengths = [len(x) for x in data]
    combined = np.concatenate(data, axis=0)
    for reducer in definition.reducers:
        combined = reducer.fit_transform(combined)
    for index, length in enumerate(lengths):
        start = sum(lengths[:index])
        end = start + length
        data[start: end] = combined[start: end]
    labels = [os.path.basename(x).split('.')[0] for x in definition.sources]
    plot_vectors(data, labels, colors, definition.output)


if __name__ == '__main__':
    main()
