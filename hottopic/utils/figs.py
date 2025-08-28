'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import singledispatch
from matplotlib import cm, colormaps, colors
from torch.nn import functional as F
from ._utils import to_list

def _format_ax(ax, title=None, aspect='equal', show_ax=True):
    if title is not None:
        ax.set_title(title)

    ax.set_aspect(aspect)

    if not show_ax:
        ax.axis('off')

    return ax

def _make_figure(n_sections=1, figsize=5, colormap=None, labels=None):
    fig, ax = plt.subplots(1, n_sections, figsize=figsize)
    axes = (ax,) if n_sections == 1 else ax

    if colormap is not None:
        cmap = colormaps.get_cmap(colormap)

        if labels is not None:
            norm = colors.Normalize(labels.min(), labels.max())

            return fig, axes, cmap, norm
        return fig, axes, cmap
    return fig, axes

def show_dataset(data, labels, sectioned=True, size=15, figsize=5, title=None, colormap='Set3', show_ax=False, show_colorbar=False, path=None, return_fig=False):
    if not sectioned:
        data = np.hstack((np.zeros((data.shape[0], 1)), data))

    sections = np.unique(data[:, 0])
    n_sections = sections.shape[0]
    title, size = to_list(n_sections, title, size)
    figsize = to_list(2, figsize*n_sections)
    fig, axes, cmap, norm = _make_figure(n_sections, figsize, colormap, labels)

    for i, a, t, s in zip(sections, axes, title, size):
        mask = data[:, 0] == i 
        a.scatter(*data[mask, 1:3].T, s=s, c=cmap(norm(labels[mask])))
        _format_ax(a, t, aspect='equal', show_ax=show_ax)

    if show_colorbar:
        fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))

    if path is not None:
        fig.savefig(path, bbox_inches='tight', transparent=True)

    if return_fig:
        return fig, axes
    else:
        plt.show()

@singledispatch
def show_comparison(data, labels, topics, title1='dataset', title2='predictions', **kwargs):
    offset = np.pad(np.ones((data.shape[0], 1)), ((0, 0), (0, data.shape[-1] - 1)))
    X, y = np.concat((data, data + offset)), np.concat((labels, topics))
    _, axes = show_dataset(X, y, return_fig=True, **kwargs)
    axes[0].set_title(title1)
    axes[1].set_title(title2)
    plt.show()

@show_comparison.register(torch.Tensor)
def _(data, labels, topics, title1='dataset', title2='predictions', **kwargs):
    offset = F.pad(torch.ones((data.shape[0], 1)), (0, data.shape[-1] - 1))
    X, y = np.concat((data, data + offset)), np.concat((labels, topics))
    _, axes = show_dataset(X, y, return_fig=True, **kwargs)
    axes[0].set_title(title1)
    axes[1].set_title(title2)
    plt.show()
