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
from sklearn.utils import check_array, check_random_state
from torch import Generator
from tqdm import tqdm
from . import _utils_

__all__ = [
    'check_data',
    'fib',
    'hello',
    'kmeans',
    'knn',
    'log_normalize',
    'normalize',
    'relabel',
    'shuffle',
    'to_list',
    'to_tensor',
    'torch_random_state'
]

fib = _utils_.fib
hello = _utils_.hello

@singledispatch
def check_data(X, accept_complex=False, accept_sparse=False, accept_large_sparse=False, dtype='numeric', order=None, ensure_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, estimator=None, input_name=''):
    check_kwargs = dict(tuple(locals().items())[2:])
    check_array_kwargs = get_kwargs(check_array, **check_kwargs)
    
    if isinstance(X, (tuple, list)):
        X = np.array(X)

    X = check_array(X, **check_array_kwargs)

    if not accept_complex and np.iscomplex(X).any():
        raise ValueError('Complex data not supported.')
    
    return X

@check_data.register(torch.Tensor)
def _(X, accept_complex=False, accept_sparse=False, accept_large_sparse=False, dtype='numeric', order=None, ensure_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, estimator=None, input_name=''):
    check_kwargs = dict(tuple(locals().items())[2:])
    check_array_kwargs = get_kwargs(check_array, **check_kwargs)
    
    if isinstance(X, (tuple, list)):
        X = np.array(X)

    X = torch.tensor(check_array(X, **check_array_kwargs))

    if not accept_complex and torch.is_complex(X):
        raise ValueError('Complex data not supported.')
    
    return X

def get_kwargs(*functions, **kwargs):
    function_kwargs = []

    for f in functions:
        keys = signature(f).parameters.keys()
        function_kwargs.append({k: kwargs[k] for k in keys if k in kwargs})

    if len(function_kwargs) == 1:
        return function_kwargs[0]
    return function_kwargs

def to_list(length, *items):
    lists = []

    for i in items:
        if isinstance(i, (tuple, list)):
            lists.append(i[:length] + i[-1:]*(length - len(i)))
        else:
            lists.append([i]*length)

    if len(lists) == 1:
        return lists[0]
    return lists

def to_tensor(*items, dtype=torch.float32):
    tensors = []

    for i in items:
        tensors.append(torch.tensor(i, dtype=dtype))

    if len(tensors) == 1:
        return tensors[0]
    return tensors

def torch_random_state(seed=None):
    if seed is None:
        return Generator()
    if isinstance(seed, Generator):
        return seed
    return Generator().manual_seed(seed)

@singledispatch
def relabel(labels, target=None):
    if target is None:
        unique, inverse = np.unique_inverse(labels)
        scores = np.eye(len(inverse))[inverse, :inverse.max() + 1]
    else:
        unique = np.unique(labels := relabel(labels))
        _, target = np.unique_inverse(target)
        scores = confusion_matrix(target[:len(labels)], labels)

    _, mask = linear_sum_assignment(scores, maximize=True)
    labels = (labels[None] == unique[mask[mask < len(unique)], None]).argmax(0)

    return labels

@relabel.register(torch.Tensor)
def _(labels, target=None):
    if target is None:
        unique, inverse = labels.unique(return_inverse=True)
        scores = torch.eye(len(inverse))[inverse, :inverse.max() + 1]
    else:
        unique = (labels := relabel(labels)).unique()
        _, target = target.unique(return_inverse=True)
        scores = confusion_matrix(target[:len(labels)], labels)

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
def kmeans(data, k=5, n_steps=100, n_perms=50, desc='KMeans', verbosity=0, seed=None):
    state, k_range = check_random_state(seed), np.arange(k)
    labels = np.zeros((n_perms, n_samples := len(data)), dtype=np.int32)

    for i in tqdm(range(n_perms), desc) if verbosity == 1 else range(n_perms):
        centroids = data[state.permutation(n_samples)[:k]]

        for _ in range(n_steps):
            labels[i] = relabel(cdist(data, centroids).argmin(-1))
            assignments = (labels[i, :, None] == k_range).astype(data.dtype)
            mask = assignments.sum(0) > 0
            assignments = assignments[:, mask]
            weights = assignments@np.diag(1/assignments.sum(0))
            centroids[mask[:k]] = weights.T@data

    labels = mode(labels).mode

    return labels

@kmeans.register(torch.Tensor)
def _(data, k=5, n_steps=100, n_perms=50, desc='KMeans', verbosity=0, seed=None):
    state, k_range = torch_random_state(seed), np.arange(k)
    labels = torch.zeros((n_perms, n_samples := len(data)), dtype=torch.int32)

    for i in tqdm(range(n_perms), desc) if verbosity == 1 else range(n_perms):
        centroids = data[torch.randperm(n_samples, generator=state)[:k]]

        for _ in range(n_steps):
            labels[i] = relabel(torch.cdist(data, centroids).argmin(-1))
            assignments = (labels[i, :, None] == k_range).to(data.dtype)
            mask = assignments.sum(0) > 0
            assignments = assignments[:, mask]
            weights = assignments@torch.diag(1/assignments.sum(0))
            centroids[mask[:k]] = weights.T@data

    labels = torch.mode(labels, 0).values

    return labels

@singledispatch
def knn(X1, X2=None, k=1, loop=True):
    if X2 is None:
        X2 = X1

    adj = cdist(X1, X2).argsort(-1)
    idx = (adj[:, :k] if loop else adj[:, 1:k + 1]).flatten()
    edges = np.vstack((np.arange(len(X1)).repeat(k), idx))

    return edges

@knn.register(torch.Tensor)
def _(X1, X2=None, k=1, loop=True):
    if X2 is None:
        X2 = X1

    adj = torch.cdist(X1, X2).argsort(-1)
    idx = (adj[:, :k] if loop else adj[:, 1:k + 1]).flatten()
    edges = torch.vstack((torch.arange(len(X1)).repeat_interleave(k), idx))

    return edges

def normalize(x):
    x = x/x.sum()

    return x

@singledispatch
def log_normalize(x):
    m, _ = x.max(1, keepdims=True)
    x = x - m - (x - m).exp().sum(1, keepdims=True).log()

    return x

@log_normalize.register(torch.Tensor)
def _(x):
    m, _ = x.max(1, keepdim=True)
    x = x - m - (x - m).exp().sum(1, keepdim=True).log()

    return x
