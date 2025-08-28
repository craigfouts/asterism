'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn
from torch.nn import functional as F

class Dirichlet(nn.Module):
    def forward(self, x, sigmoid=True):
        if sigmoid:
            x = F.sigmoid(x)

        products = F.pad((1 - x).cumprod(-1), (1, 0), value=1)
        weights = F.pad(x, (0, 1), value=1)*products

        return weights
