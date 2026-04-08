'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn, optim
from torch.nn import functional as F

__all__ = [
    'ACTS',
    'NORMS',
    'OPTIMS',
    'Dirichlet'
]

class Dirichlet(nn.Module):
    def forward(self, X, sigmoid=True):
        if sigmoid:
            X = F.sigmoid(X)

        products = F.pad((1 - X).cumprod(-1), (1, 0), value=1)
        weights = F.pad(X, (0, 1), value=1)*products

        return weights

ACTS = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus, 'softmax': nn.Softmax, 'dirichlet': Dirichlet}
NORMS = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
OPTIMS = {'adam': optim.Adam, 'sgd': optim.SGD}
