'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state
from . import to_list, to_tensor

CHECKERS = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=np.int32)

POLYGONS = np.array([
    [0, 1, 0, 2, 2, 2],
    [1, 1, 1, 2, 0, 2],
    [0, 1, 0, 2, 2, 2],
    [3, 0, 0, 4, 4, 4],
    [3, 3, 0, 0, 4, 0],
    [3, 3, 3, 0, 4, 0]
], dtype=np.int32)

_TEMPLATES = {'checkers': CHECKERS, 'polygons': POLYGONS}

def _make_blocks(template='polygons', block_size=5):
    n_sections = len(template) if isinstance(template, (tuple, list)) else 1
    template, block_size = to_list(n_sections, template, block_size)
    template = [_TEMPLATES[t] if isinstance(t, str) else t for t in template]
    locs, labels = [], []

    for i, (t, s) in enumerate(zip(template, block_size)):
        grid = np.repeat(np.repeat(t, s, 0), s, 1)[::-1, :, None]
        topics, counts = np.unique(grid, return_counts=True)
        locs += [np.vstack(np.where(grid.T == t)).T + (i, 0., 0.) for t in topics]
        labels.append(np.repeat(topics, counts))

    locs, labels = np.vstack(locs), np.hstack(labels)

    return locs, labels

def _make_data(template='polygons', block_size=10, n_features=100, n_equivocal=0, n_redundant=0, n_repeated=0, topic_sep=10., scale=1., seed=None):
    state = check_random_state(seed)
    locs, labels = _make_blocks(template, block_size)
    n_samples, (_, bins) = len(locs), np.unique(labels, return_counts=True)
    n_topics, weights = len(bins), bins/n_samples
    n_informative = n_features - n_equivocal - n_redundant - n_repeated
    features, topics = make_classification(n_samples, n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_topics, n_clusters_per_class=1, weights=weights, flip_y=0., class_sep=topic_sep, scale=scale, shuffle=False, random_state=state)
    data = features[topics.argsort()[labels.argsort().argsort()]]

    return data, locs, labels

def make_dataset(template='polygons', block_size=10, n_features=100, n_equivocal=0, n_redundant=0, n_repeated=0, topic_sep=10., scale=1., wiggle=0., mix=0., return_tensor=False, seed=None):
    state = check_random_state(seed)
    data, locs, labels = _make_data(template, block_size, n_features, n_equivocal, n_redundant, n_repeated, topic_sep, scale, state)
    sections, locs[:, 1:] = np.unique(locs[:, 0]), state.normal(locs[:, 1:], wiggle)

    for i in sections:
        mask = locs[:, 0] == i
        n_samples, offset = mask.sum(), mask.argmax()
        idx = state.permutation(n_samples)[:int(mix*n_samples)] + offset
        locs[idx, 1:] = state.permutation(locs[idx, 1:])

    if return_tensor:
        return *to_tensor(data, locs), to_tensor(labels, dtype=torch.int32)
    return data, locs, labels
