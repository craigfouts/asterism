'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils import get_kwargs, torch_random_state
from ..utils.nets import ACTS, NORMS
from ..utils.sugar import attrmethod, buildmethod, checkmethod

__all__ = [
    'Encoder',
    'MLP',
    'RNN',
    'VAE'
]

class MLP(nn.Sequential):
    @attrmethod
    def __init__(self, *channels, bias=True, norm_layer=None, act_layer=None, dropout=0., final_bias=True, final_norm=None, final_act=None, final_dropout=0., **kwargs):
        modules = []

        for i in range(1, len(channels) - 1):
            modules.append(self.layer(channels[i - 1], channels[i], bias, norm_layer, act_layer, dropout, **kwargs))
        
        modules.append(self.layer(channels[-2%len(channels)], channels[-1], final_bias, final_norm, final_act, final_dropout, **kwargs))
        super().__init__(*modules)

    @staticmethod
    def layer(in_channels, out_channels=None, bias=True, norm_layer=None, act_layer=None, dropout=0., **kwargs):
        if out_channels is None:
            out_channels = in_channels

        layer_kwargs = dict(tuple(locals().items())[2:-1], **kwargs)
        modules = [nn.Linear(in_channels, out_channels, bias)]

        if norm_layer is not None:
            norm_kwargs = get_kwargs(norm := NORMS[norm_layer], **layer_kwargs)
            modules.append(norm(out_channels, **norm_kwargs))

        if act_layer is not None:
            act_kwargs = get_kwargs(act := ACTS[act_layer], **layer_kwargs)
            modules.append(act(**act_kwargs))

        if dropout > 0.:
            modules.append(nn.Dropout(dropout))

        module = nn.Sequential(*modules)

        return module

class RNN(MLP):
    @attrmethod
    def __init__(self, channels, bias=True, norm_layer=None, act_layer='tanh', dropout=0., seed=None, **kwargs):
        super().__init__(channels, final_bias=bias, final_norm=norm_layer, final_act=act_layer, final_dropout=dropout, **kwargs)

        self._state = torch_random_state(seed)
        self.X_ = torch.rand(1, channels, generator=self._state)

    def forward(self, X=None, n_layers=2):
        if X is None:
            X = self.X_

        for i in range(1, n_layers):
            X = torch.cat((X, super().forward(X[i - 1:i])))

        return X
    
class Encoder(nn.Module):
    @attrmethod
    def __init__(self, *channels, bias=True, norm_layer='batch', act_layer='relu', dropout=.5, seed=None, **kwargs):
        super().__init__()

        self._state = torch_random_state(seed)
        self._channels = channels if len(channels) > 2 else (channels[0], (channels[0] + channels[-1])//2, channels[-1])
        self._q_net = MLP(*self._channels[:-1], norm_layer=norm_layer, act_layer=act_layer, dropout=dropout, final_norm=norm_layer, final_act=act_layer, final_dropout=dropout, **kwargs)
        self._m_mlp = MLP(*self._channels[-2:], final_bias=bias, final_norm=norm_layer, **kwargs)
        self._s_mlp = MLP(*self._channels[-2:], final_bias=bias, **kwargs)

    def forward(self, X, return_kld=False):
        q = self._q_net(X)
        m, s_log = self._m_mlp(q), self._s_mlp(q)
        s_exp = (.5*s_log).exp()
        z = m + s_exp*torch.randn(m.shape, generator=self._state)

        if return_kld:
            kld = (m**2 + s_exp**2 - s_log - .5).sum()

            return z, kld
        return z

class VAE(BaseEstimator, TransformerMixin, nn.Module):
    @attrmethod
    def __init__(self, *channels, bias=True, norm_layer='batch', act_layer='relu', dropout=.5, kld_scale=.1, optim='adam', desc='VAE'):
        super().__init__()

        self.channels = channels
        self.bias = bias
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.dropout = dropout
        self.kld_scale = kld_scale
        self.optim = optim
        self.desc = desc

        self._channels = channels if len(channels) > 0 else (64, 32)
        self._step_n = 0

    def __call__(self, X, eval=True):
        Z = self.transform(X, eval)

        return Z

    def _build(self, X, learning_rate=1e-2, batch_size=32, shuffle=True):
        if batch_size is -1:
            batch_size = X.shape[0]

        self._loader = DataLoader(X, batch_size, shuffle)
        self._encoder = Encoder(X.shape[1], *self._channels, mode=self.mode, bias=self.bias, norm_layer=self.norm_layer, act_layer=self.act_layer, dropout=self.dropout)
        self._decoder = MLP(*self._channels[::-1], X.shape[1], mode=self.mode, bias=self.bias, norm_layer=self.norm_layer, act_layer=self.act_layer, dropout=self.dropout)
        self._optim = OPTIMS[self.optim](self.parameters(), lr=learning_rate)
        self.train()

        return self
    
    def _step(self):
        loss = 0.

        for X in self._loader:
            Z, kld = self._encoder(X, return_kld=True)
            X_ = self._decoder(Z)
            X_loss = (X_ - X).square().sum().sqrt() + self.kld_scale*kld
            X_loss.backward()
            loss += X_loss.item()

        self._optim.step()
        self._optim.zero_grad()

        return loss
    
    def _display(self):
        desc = self.desc + '  ' if self.desc is not None else ''
        msg = f'{desc}step: {self._step_n}'

        if hasattr(self, 'log_'):
            msg += f'  score: {self.log_[-1]}'

        print(msg)

    @checkmethod
    @buildmethod
    def fit(self, X, n_steps=200, verbosity=1, display_rate=10, **kwargs):
        fit_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        step_kwargs, display_kwargs = get_kwargs(self._step, self._display, **fit_kwargs)
        self.log_ = []

        for self._step_n in tqdm(range(n_steps), self.desc) if verbosity == 1 else range(n_steps):
            self.log_.append(self._step(**step_kwargs))

            if verbosity == 2 and self._step%display_rate == 0:
                self._display(**display_kwargs)

        return self
    
    def transform(self, X, eval=True):
        if eval:
            self.eval()

        Z = self._encoder(X).detach()

        return Z

