import pickle
import os
import definitions
from definitions.attrdict import AttrDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        source.uuids = os.path.join(ROOT, source.uuids)
        definition.sources[index] = source
    return definition


def collect_articles(definition, embedding, uuids):
    corpus = np.load(os.path.join(definition.directory, definition.uuids))
    indices = [np.argmax(corpus == x) for x in uuids]
    return embedding[indices]


def plot_vectors(data, uuids, sources, output):
    fig, ax = plt.subplots(figsize=(12, 8))
    handles = []
    colors = {x: 'gray' for x in uuids}
    for source in sources:
        for uuid in open(source.uuids).read().split():
            colors[uuid] = source.color
        handles.append(mpatches.Patch(color=source.color, label=source.label))
    colors = [colors[x] for x in uuids]
    ax.scatter(data[:, 0], data[:, 1], c=colors)
    plt.legend(handles=handles, loc='upper right')
    ensure_directory(os.path.dirname(output))
    fig.savefig(output, dpi=300)


def main():
    definition = load_definition()
    data = np.load(os.path.join(ROOT, definition.embedding))
    uuids = np.load(os.path.join(ROOT, definition.uuids))

    pca = KernelPCA(**definition.pca)
    tsne = TSNE(**definition.tsne)
    data = pca.fit_transform(data)
    data = tsne.fit_transform(data)

    plot_vectors(data, uuids, definition.sources, definition.output)


if __name__ == '__main__':
    main()
