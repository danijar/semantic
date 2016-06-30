import itertools
import pickle
import os
import definitions
import numpy as np
from sklearn.cross_validation import KFold
from semantic.utility import ensure_directory


ROOT = os.path.join(os.path.dirname(__file__), '..')


def load_definition():
    schema = os.path.join(ROOT, 'schema/train.yaml')
    definition = os.path.join(ROOT, 'definition/train.yaml')
    definition = definitions.Parser(schema)(definition)
    definition.directory = os.path.join(ROOT, definition.directory)
    ensure_directory(definition.directory)
    return definition


def load_users(definition):
    filepath = os.path.join(ROOT, definition.users)
    with open(filepath) as file_:
        for line in file_:
            words = line.strip().split()
            user_id = words[0]
            uuids = words[1:]
            yield user_id, uuids


def collect_articles(definition, embedding, uuids):
    corpus = np.load(os.path.join(definition.directory, definition.uuids))
    indices = [np.argmax(corpus == x) for x in uuids]
    return embedding[indices]


def collect_remaining_articles(definition, embedding, uuids):
    corpus = np.load(os.path.join(definition.directory, definition.uuids))
    indices = [np.argmax(corpus == x) for x in uuids]
    neg_indices = list(set(range(len(uuids))) - set(indices))
    return embedding[neg_indices]


def load_embeddings(definition):
    for filename in definition.embeddings:
        print('Load embedding', filename)
        filepath = os.path.join(ROOT, definition.directory, filename)
        with open(filepath, 'rb') as file_:
            embedding = np.load(file_)
        name, _ = os.path.splitext(filename)
        yield name, embedding


def nearest_neighbor(samples, embedding):
    nearest = np.empty(samples.shape)
    for index, sample in enumerate(samples):
        distances = ((embedding - sample) ** 2).sum(axis=1)
        closest = np.argmin(distances, axis=0)
        nearest[index] = embedding[closest]
    assert len(nearest) == len(samples)
    return nearest


def training(distribution, positives, embedding, folds, oversample=10):
    """Return a list of log probs for each document."""
    folds = KFold(
        positives.shape[0], n_folds=folds, shuffle=True, random_state=0)
    for train, test in folds:
        train, test = positives[train], positives[test]
        distribution.fit(train)
        samples = distribution._model.sample(oversample * len(test), 42)
        samples = nearest_neighbor(samples, embedding)
        combined = np.concatenate([test, samples], axis=0)
        hits = len(combined) - len({tuple(x) for x in combined.tolist()})
        yield hits, len(test)


def evaluation(definition, distribution, embedding, name):
    hits, amount = 0, 0
    for user_id, uuids in load_users(definition):
        positives = collect_articles(definition, embedding, uuids)
        runs = training(distribution, positives, embedding, definition.folds)
        for h, a in runs:
            hits += h
            amount += a
    message = '{:<12} error {:6.2f}% after 10x sampling'
    print(hits, amount)
    error = 1 - (hits / amount)
    print(message.format(name, 100 * error))


def store_distribution(definition, distribution, embedding, name):
    directory = os.path.join(definition.directory, 'users')
    ensure_directory(directory)
    for user_id, uuids in load_users(definition):
        data = collect_articles(definition, embedding, uuids)
        distribution.fit(data)
        filename = '{}-{}.pkl'.format(name, user_id)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as file_:
            pickle.dump(distribution, file_)


def main():
    definition = load_definition()
    embeddings = load_embeddings(definition)
    combinations = itertools.product(embeddings, definition.distributions)
    for (emb_name, embedding), distribution in combinations:
        dist_name = type(distribution).__name__
        name = '{}-{}'.format(emb_name, dist_name)
        evaluation(definition, distribution, embedding, name)
        store_distribution(definition, distribution, embedding, name)


if __name__ == '__main__':
    main()
