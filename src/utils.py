"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import singledispatch
from matplotlib import cm, colormaps, colors
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

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

def format_ax(ax, title=None, aspect='equal', show_ax=True):
    if title is not None:
        ax.set_title(title)

    ax.set_aspect(aspect)

    if not show_ax:
        ax.axis('off')

    return ax

def make_figure(n_sections=1, figsize=5, colormap=None, labels=None):
    fig, ax = plt.subplots(1, n_sections, figsize=figsize)
    axes = (ax,) if n_sections == 1 else ax

    if colormap is not None:
        cmap = colormaps.get_cmap(colormap)

        if labels is not None:
            norm = colors.Normalize(labels.min(), labels.max())

            return fig, axes, cmap, norm
        return fig, axes, cmap
    return fig, axes

def show_dataset(data, labels, sectioned=True, size=15, figsize=5, title=None, colormap='Set3', show_ax=False, show_colorbar=False, path=None):
    if not sectioned:
        data = np.hstack((np.zeros((data.shape[0], 1)), data))

    sections = np.unique(data[:, 0])
    n_sections = sections.shape[0]
    title, size = to_list(n_sections, title, size)
    figsize = to_list(2, figsize*n_sections)
    fig, axes, cmap, norm = make_figure(n_sections, figsize, colormap, labels)

    for i, a, t, s in zip(sections, axes, title, size):
        mask = data[:, 0] == i 
        a.scatter(*data[mask, 1:3].T, s=s, c=cmap(norm(labels[mask])))
        format_ax(a, t, aspect='equal', show_ax=show_ax)

    if show_colorbar:
        fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))

    if path is not None:
        fig.savefig(path, bbox_inches='tight', transparent=True)
        