"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import numpy as np
from sklearn.datasets import make_classification
import torch
from utils import itemize

CHECKERS = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=np.int32)

POLYGONS = np.array([[0, 1, 0, 2, 2, 2],
                     [1, 1, 1, 2, 0, 2],
                     [0, 1, 0, 2, 2, 2],
                     [3, 0, 0, 4, 4, 4],
                     [3, 3, 0, 0, 4, 0],
                     [3, 3, 3, 0, 4, 0]], dtype=np.int32)

TEMPLATES = {'checkers': CHECKERS, 'polygons': POLYGONS}

def make_blocks(template='polygons', block_size=5):
    """Generates sample locations and labels based on the given block template.
    
    Parameters
    ----------
    template : str | ndarray | tuple | list, default='polygons'
        Template name(s) or array(s).
    block_size : int | tuple | list, default=5
        Size(s) of each block.

    Returns
    -------
    ndarray
        Sample locations.
    ndarray
        Sample labels.
    """

    n_sections = len(template) if isinstance(template, (tuple, list)) else 1
    template, block_size = itemize(n_sections, template, block_size)
    template = [TEMPLATES[block] if isinstance(block, str) else block for block in template]
    locs, labels = [], []

    for i, (t, s) in enumerate(zip(template, block_size)):
        grid = np.repeat(np.repeat(t, s, 0), s, 1)[::-1, :, None]
        topics, counts = np.unique(grid, return_counts=True)
        locs += [np.stack(np.where(grid.T == topic)).T + (i, 0., 0.) for topic in topics]
        labels.append(np.repeat(topics, counts))

    locs, labels = np.vstack(locs), np.hstack(labels)

    return locs, labels

def make_data(template='polygons', block_size=10, n_features=100, n_equivocal=0, n_redundant=0, n_repeated=0, topic_sep=10., scale=1.):
    """Generates sample locations, labels, and features based on the given block
    template and feature parameterization.
    
    Parameters
    ----------
    template : str | ndarray | tuple | list, default='polygons
        Template name(s) or array(s).
    block_size : int | tuple | list, default=5
        Size(s) of each block.
    n_features : int, default=100
        Number of features per sample.
    n_equivocal : int, default=0
        Number of equivocal features.
    n_redundant : int, default=0
        Number of redundant features.
    n_repeated : int, default=0
        Number of repeated features.
    topic_sep : float, default=10.0
        Scale of feature separation between topics.
    scale : float, default=1.0
        Scale of feature values.

    Returns
    -------
    ndarray
        Sample locations and features.
    ndarray
        Sample labels.
    """
    
    locs, labels = make_blocks(template, block_size)
    n_samples = locs.shape[0]
    _, bins = np.unique(labels, return_counts=True)
    n_topics, weights = bins.shape[0], bins/n_samples
    n_informative = n_features - n_equivocal - n_redundant - n_repeated
    features, topics = make_classification(n_samples, n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_topics, n_clusters_per_class=1, weights=weights, flip_y=0., class_sep=topic_sep, scale=scale, shuffle=False)
    data = np.hstack([locs, features[topics.argsort()[labels.argsort().argsort()]]])

    return data, labels

def make_dataset(template='polygons', block_size=10, n_features=100, n_equivocal=0, n_redundant=0, n_repeated=0, topic_sep=10., scale=1., wiggle=0., mix=0., return_tensor=False):
    """Generates a sample dataset based on the given block template and feature
    parameterization.
    
    Parameters
    ----------
    template : str | ndarray | tuple | list, default='polygons'
        Template name(s) or array(s).s
    block_size : int | tuple | list, default=5
        Size(s) of each block.
    n_features : int, default=100
        Number of features per sample.
    n_equivocal : int, default=0
        Number of equivocal features.
    n_redundant : int, default=0
        Number of redundant features.
    n_repeated, default=0
        Number of repeated features.
    topic_sep : float, default=10.0
        Feature separation factor.
    scale : float, default=1.0
        Feature scale factor.

    Returns
    -------
    ndarray | tensor
        Sample dataset.
    ndarray | tensor
        Sample labels.
    """

    data, labels = make_data(template, block_size, n_features, n_equivocal, n_redundant, n_repeated, topic_sep, scale)
    data[:, 1:3] = np.random.normal(data[:, 1:3], wiggle)
    sections = np.unique(data[:, 0])

    for i, s in enumerate(sections):
        mask = data[:, 0] == s
        n_samples, offset = mask.sum(), mask.argmax()
        idx = np.random.permutation(n_samples)[:int(mix*n_samples)] + offset
        data[idx, 1:3] = np.random.permutation(data[idx, 1:3])

    if return_tensor:
        data, labels = torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32)

    return data, labels
