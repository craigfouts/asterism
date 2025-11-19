'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn
from torch.nn import functional as F

__all__ = [
    'Dirichlet'
]

class Dirichlet(nn.Module):
    def forward(self, X, sigmoid=True):
        if sigmoid:
            X = F.sigmoid(X)

        products = F.pad((1 - X).cumprod(-1), (1, 0), value=1)
        weights = F.pad(X, (0, 1), value=1)*products

        return weights
