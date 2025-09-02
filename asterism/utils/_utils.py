'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
import torch
from functools import singledispatch
from inspect import signature
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def get_kwargs(*func, **kwargs):
    func_kwargs = []

    for f in func:
        keys = signature(f).parameters.keys()
        func_kwargs.append({key:kwargs[key] for key in keys if key in kwargs})

    if len(func_kwargs) == 1:
        return func_kwargs[0]
    return func_kwargs

def to_list(length, *items):
    lists = []

    for item in items:
        if isinstance(item, (tuple, list)):
            lists.append(item[:length] + item[-1:]*(length - len(item)))
        else:
            lists.append([item]*length)

    if len(lists) == 1:
        return lists[0]
    return lists

@singledispatch
def relabel(labels, target=None):
    unique, inverse = np.unique_inverse(labels)

    if target is None:
        scores = np.eye(inverse.shape[0])[inverse, :inverse.max() + 1]
    else:
        labels, target = relabel(labels), np.unique_inverse(target)[-1]
        unique, inverse = np.unique_inverse(labels)
        scores = confusion_matrix(target[:labels.shape[0]], labels)

    _, mask = linear_sum_assignment(scores, maximize=True)
    labels = (labels[None] == unique[mask[mask < len(unique)], None]).argmax(0)

    return labels

@relabel.register(torch.Tensor)
def _(labels, target=None):
    unique, inverse = labels.unique(return_inverse=True)

    if target is None:
        scores = torch.eye(inverse.shape[0])[inverse, :inverse.max() + 1]
    else:
        labels, target = relabel(labels), target.unique(return_inverse=True)[-1]
        unique, inverse = labels.unique(return_inverse=True)
        scores = confusion_matrix(target[:labels.shape[0]], labels)

    _, mask = linear_sum_assignment(scores, maximize=True)
    labels = (labels[None] == unique[mask[mask < len(unique)], None]).float().argmax(0)

    return labels

def shuffle(data, labels=None, sort=False, cut=None):
    mask = np.random.permutation(data.shape[-2])[:cut]
    data = data[:, mask] if data.ndim > 2 else data[mask]

    if labels is not None:
        labels = relabel(labels[mask]) if sort else labels[mask]

        return data, labels
    return data

@singledispatch
def kmeans(data, k=3, n_steps=10, n_perms=100, verbosity=1, desc='KMeans'):
    labels, k_range = np.zeros((n_perms, data.shape[0]), dtype=np.int32), np.arange(k)

    for i in tqdm(range(n_perms), desc) if verbosity == 1 else range(n_perms):
        centroids = data[np.random.permutation(data.shape[0])[:k]]

        for _ in range(n_steps):
            labels[i] = relabel(cdist(data, centroids).argmin(-1))
            assignments = (labels[i][:, None] == k_range).astype(data.dtype)
            mask = assignments.sum(0) > 0
            assignments = assignments[:, mask]
            weights = assignments@np.diag(1/assignments.sum(0))
            centroids[mask[:centroids.shape[0]]] = weights.T@data

    labels = mode(labels).mode

    return labels

@kmeans.register(torch.Tensor)
def _(data, k=3, n_steps=10, n_perms=100, verbosity=1, desc='KMeans'):
    labels, k_range = torch.zeros(n_perms, data.shape[0], dtype=torch.int32), torch.arange(k)

    for i in tqdm(range(n_perms), desc) if verbosity == 1 else range(n_perms):
        centroids = data[torch.randperm(data.shape[0])[:k]]

        for _ in range(n_steps):
            labels[i] = relabel(torch.cdist(data, centroids).argmin(-1))
            assignments = (labels[i][:, None] == k_range).to(data.dtype)
            mask = assignments.sum(0) > 0
            assignments = assignments[:, mask]
            weights = assignments@torch.diag(1/assignments.sum(0))
            centroids[mask[:centroids.shape[0]]] = weights.T@data

    labels = torch.mode(labels, 0).values

    return labels
        