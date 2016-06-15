import pickle
import os
import definitions
from definitions.attrdict import AttrDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from semantic.utility import ensure_directory


ROOT = os.path.join(os.path.dirname(__file__), '..')


def load_definition():
    schema = os.path.join(ROOT, 'schema/visualize.yaml')
    definition = os.path.join(ROOT, 'definition/visualize.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.output = os.path.join(ROOT, definition.output)
    for index, source in enumerate(definition.sources):
        source = AttrDict(source)
        source.filename = os.path.join(ROOT, source.filename)
        definition.sources[index] = source
    return definition


def load_data(definition):
    with open(os.path.join(ROOT, definition.vectorizer), 'rb') as file_:
        vectorizer = pickle.load(file_)
    data, colors = [], []
    for source in definition.sources:
        _, vectors = vectorizer.transform(source.filename)
        data.append(vectors)
        colors.append(source.color)
    return data, colors


def plot_vectors(data, labels, colors, output):
    fig, ax = plt.subplots(figsize=(12, 8))
    for points, label, color in zip(data, labels, colors):
        ax.scatter(points[:, 0], points[:, 1], label=label, c=color)
    ax.legend(loc='upper right')
    ensure_directory(output)
    filename = os.path.join(output, 'figure.png')
    fig.savefig(filename, dpi=300)


def main():
    definition = load_definition()
    data, colors = load_data(definition)
    lengths = [len(x) for x in data]

    pca = KernelPCA(**definition.pca)
    tsne = TSNE(**definition.tsne)
    combined = np.concatenate(data, axis=0)
    combined = pca.fit_transform(combined)
    combined = tsne.fit_transform(combined)

    data = []
    for index, length in enumerate(lengths):
        index = sum(lengths[:index])
        data.append(combined[index: index + length])
    labels = [os.path.basename(x.filename).split('.')[0]
              for x in definition.sources]
    plot_vectors(data, labels, colors, definition.output)


if __name__ == '__main__':
    main()
