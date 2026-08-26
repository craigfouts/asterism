'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
import re
import torch
import torch.nn.functional as F
from functools import singledispatch
from inspect import signature
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_array, check_random_state
from torch import Generator
from tqdm import tqdm
# from . import _utils_

__all__ = [
    # 'cdist',
    'check_data',
    'get_kwargs',
    'get_methods',
    'kmeans',
    'knn',
    'knn2D',
    'log_normalize',
    'normalize',
    'pad',
    'random_state',
    'relabel',
    'shuffle',
    'batch_split',
    'to_list',
    'to_tensor',
    'torch_random_state',
    'fps'
]

# cdist = _utils_.cdist

def torch_random_state(seed=None):
    if seed is None:
        return Generator()
    if isinstance(seed, Generator):
        return seed
    return Generator().manual_seed(seed)

def random_state(seed=None, torch_state=False):
    if torch_state:
        return torch_random_state(seed)
    return check_random_state(seed)

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

def get_methods(cls, prefix='', suffix='', return_callable=False):
    methods = []

    for attr in dir(cls):
        if callable(method := getattr(cls, attr)) and re.search(f'^{prefix}.*{suffix}$', attr):
            methods.append(method if return_callable else attr)

    return methods

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

@singledispatch
def pad(X, pad):
    out = np.pad(X, pad)

    return out

@pad.register(torch.Tensor)
def _(X, pad):
    out = F.pad(X, pad)

    return out

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
def batch_split(x, y, imgs=None, n_test=1):
    if imgs is None:
        return x[None], y[None]

    if imgs.ndim > 1:
        imgs = imgs[:, 0]

    n_train = np.unique(imgs).shape[0] - n_test
    train_size = x.shape[0] - (test_size := (imgs >= n_train).sum())
    x_train = x[:train_size].reshape((n_train, train_size//n_train, x.shape[-1]))
    x_test = x[train_size:].reshape((n_test, test_size//n_test, x.shape[-1]))
    y_train, y_test = y[:train_size], y[train_size:]

    return x_train, y_train, x_test, y_test

@batch_split.register(torch.Tensor)
def _(x, y, imgs=None, n_test=1):
    if imgs is None:
        return x[None], y[None]
    
    if imgs.ndim > 1:
        imgs = imgs[:, 0]

    n_train = imgs.unique().shape[0] - n_test
    train_size = x.shape[0] - (test_size := (imgs >= n_train).sum())
    x_train = x[:train_size].view([n_train, train_size//n_train, x.shape[-1]])
    x_test = x[train_size:].view([n_test, test_size//n_test, x.shape[-1]])
    y_train, y_test = y[:train_size], y[train_size:]

    return x_train, y_train, x_test, y_test

@singledispatch
def knn(x, k=1, loop=True):
    adj = cdist(x, x).argsort(-1)
    idx = (adj[:, :k] if loop else adj[:, 1:k + 1]).flatten()
    edges = np.vstack((np.arange(len(X)).repeat(k), idx))

    return edges

@knn.register(torch.Tensor)
def _(x, k=1, loop=True):
    adj = torch.cdist(x, x).argsort(-1)
    idx = (adj[:, :k] if loop else adj[:, 1:k + 1]).flatten()
    edges = torch.vstack((torch.arange(x.shape[0]).repeat_interleave(k), idx))

    return edges

@singledispatch
def knn2D(X, k=1, loop=True):
    X = pad(X, ((n := 3 - X.shape[1])*(n > 0), 0))
    edges = np.zeros(2, len(X)*k, dtype=np.int32)

    for i in range(len(np.unique(X[:, 0]))):
        mask_i, mask_h = X[:, 0] == i, X[:, 0] < i
        end = (start := (m := mask_h.sum())*k) + mask_i.sum()*k
        edges[:, start:end] = knn(X[mask_i], k) + m

    return edges

@knn2D.register(torch.Tensor)
def _(X, k=1, loop=True):
    X = pad(X, ((n := 3 - X.shape[1])*(n > 0), 0))
    edges = torch.zeros(2, len(X)*k, dtype=torch.int32)

    for i in range(len(X[:, 0].unique())):
        mask_i, mask_h = X[:, 0] == i, X[:, 0] < i
        end = (start := (m := mask_h.sum())*k) + mask_i.sum()*k
        edges[:, start:end] = knn(X[mask_i], k) + m

    return edges

@singledispatch
def fps(x, n_samples=5, seed=None, return_idx=False):
    state = check_random_state(seed)
    idx = np.empty(n_samples, dtype=np.int32)
    idx[0] = state.randint(n_pts := x.shape[0])
    prox = np.full(n_pts, np.inf)

    for i in range(1, n_samples):
        prox_ = np.linalg.norm(x - x[idx[i - 1]], axis=-1)
        idx[i] = (prox := np.minimum(prox, prox_)).argmax()

    samples = idx if return_idx else x[idx]

    return samples

@fps.register(torch.Tensor)
def _(x, n_samples=5, seed=None, return_idx=False):
    state = torch_random_state(seed)
    idx = torch.empty(n_samples, dtype=torch.int32)
    idx[0] = torch.randint(n_pts := x.shape[0], (1,), generator=state)
    prox = torch.full((n_pts,), torch.inf)

    for i in range(1, n_samples):
        prox_ = torch.linalg.norm(x - x[idx[i - 1]], dim=-1)
        idx[i] = (prox := torch.minimum(prox, prox_)).argmax()

    samples = idx if return_idx else x[idx]

    return samples

@singledispatch
def kmeans(data, k=5, n_steps=100, n_perms=10, desc='KMeans', verbosity=0, seed=None):
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
def _(data, k=5, n_steps=100, n_perms=10, desc='KMeans', verbosity=0, seed=None):
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

def normalize(x):
    x /= x.sum()

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
