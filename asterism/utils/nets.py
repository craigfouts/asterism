'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from torch import nn, optim
from .acts import Dirichlet
from ._utils import get_kwargs

NORM = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
ACT = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus, 'softmax': nn.Softmax, 'dirichlet': Dirichlet}
OPTIM = {'adam': optim.Adam, 'sgd': optim.SGD}

class MLP(nn.Sequential):
    def __init__(self, *channels, bias=True, norm_layer=None, act_layer=None, dropout=0., final_bias=True, final_norm=None, final_act=None, final_dropout=0., **kwargs):        
        modules = []

        for i in range(1, len(channels) - 1):
            modules.append(self._layer(channels[i - 1], channels[i], bias, norm_layer, act_layer, dropout, **kwargs))

        modules.append(self._layer(channels[-2%len(channels)], channels[-1], final_bias, final_norm, final_act, final_dropout, **kwargs))
        super().__init__(*modules)

    @staticmethod
    def _layer(in_channels, out_channels=None, bias=True, norm_layer=None, act_layer=None, dropout=0., **kwargs):
        if out_channels is None:
            out_channels = in_channels

        layer_kwargs = dict(tuple(locals().items())[2:-1], **kwargs)
        modules = [nn.Linear(in_channels, out_channels, bias)]

        if norm_layer is not None:
            norm_kwargs = get_kwargs(NORM[norm_layer], **layer_kwargs)
            modules.append(NORM[norm_layer](out_channels, **norm_kwargs))

        if act_layer is not None:
            act_kwargs = get_kwargs(ACT[act_layer], **layer_kwargs)
            modules.append(ACT[act_layer](**act_kwargs))

        if dropout > 0.:
            modules.append(nn.Dropout(dropout))

        module = nn.Sequential(*modules)

        return module
    
class RNN(MLP):
    def __init__(self, channels, bias=True, norm_layer=None, act_layer='tanh', dropout=0., **kwargs):
        super().__init__(channels, final_bias=bias, final_norm=norm_layer, final_act=act_layer, final_dropout=dropout, **kwargs)

        self.x_ = torch.rand(1, channels)

    def forward(self, x=None, n_layers=2):
        if x is None:
            x = self.x_

        for i in range(1, n_layers):
            x = torch.cat([x, super().forward(x[i - 1:i])])

        return x
    
class Encoder(nn.Module):
    def __init__(self, *channels, bias=True, norm_layer='batch', act_layer='relu', dropout=.5, **kwargs):
        super().__init__()

        self.channels = channels if len(channels) > 2 else (channels[0], (channels[0] + channels[-1])//2, channels[-1])

        self._q_net = MLP(*self.channels[:-1], norm_layer=norm_layer, act_layer=act_layer, dropout=dropout, final_norm=norm_layer, final_act=act_layer, final_drop=dropout, **kwargs)
        self._m_mlp = MLP(*self.channels[-2:], final_bias=bias, final_norm=norm_layer, **kwargs)
        self._s_mlp = MLP(*self.channels[-2:], final_bias=bias, **kwargs)


    def forward(self, X, return_kld=False):
        q = self._q_net(X)
        m, s_log = self._m_mlp(q), self._s_mlp(q)
        s_exp = (.5*s_log).exp()
        z = m + s_exp*torch.randn_like(m)

        if return_kld:
            kld = (m**2 + s_exp**2 - s_log - .5).sum()

            return z, kld
        return z
        