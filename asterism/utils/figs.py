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

def _make_plot(n_sections=1, fig_size=5, colormap=None, labels=None):
    fig_size = to_list(2, n_sections*fig_size)
    fig, ax = plt.subplots(1, n_sections, figsize=fig_size)
    axes = (ax,) if n_sections == 1 else ax

    if colormap is not None:
        cmap = colormaps.get_cmap(colormap)

        if labels is not None:
            norm = colors.Normalize(labels.min(), labels.max())

            return fig, axes, cmap, norm
        return fig, axes, cmap
    return fig, axes

def show_dataset(locs, labels, size=15, fig_size=5, title=None, colormap='Set3', show_ax=False, show_colorbar=False, path=None, return_fig=False):
    if locs.shape[1] < 3:
        locs = np.hstack((np.zeros((locs.shape[0], 1)), locs))

    sections = np.unique(locs[:, 0])
    n_sections = sections.shape[0]
    title, size = to_list(n_sections, title, size)
    fig, axes, cmap, norm = _make_plot(n_sections, fig_size, colormap, labels)

    for i, a, t, s in zip(sections, axes, title, size):
        mask = locs[:, 0] == i 
        a.scatter(*locs[mask, 1:].T, s=s, c=cmap(norm(labels[mask])))
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
def show_comparison(locs, labels, topics, title1='dataset', title2='predictions', **kwargs):
    offset = np.pad(np.ones((locs.shape[0], 1)), ((0, 0), (0, locs.shape[-1] - 1)))
    X, y = np.concat((locs, locs + offset)), np.concat((labels, topics))
    _, axes = show_dataset(X, y, return_fig=True, **kwargs)
    axes[0].set_title(title1)
    axes[1].set_title(title2)
    plt.show()

@show_comparison.register(torch.Tensor)
def _(locs, labels, topics, title1='dataset', title2='predictions', **kwargs):
    offset = F.pad(torch.ones((locs.shape[0], 1)), (0, locs.shape[-1] - 1))
    X, y = np.concat((locs, locs + offset)), np.concat((labels, topics))
    _, axes = show_dataset(X, y, return_fig=True, **kwargs)
    axes[0].set_title(title1)
    axes[1].set_title(title2)
    plt.show()
