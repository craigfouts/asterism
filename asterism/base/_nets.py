'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils import get_kwargs, torch_random_state
from ..utils.nets import ACTS, NORMS, OPTIMS
from ..utils.sugar import attrmethod, buildmethod, checkmethod

__all__ = [
    'Encoder',
    'MLP',
    'RNN',
    'VAE'
]

class MLP(nn.Sequential):
    @attrmethod
    def __init__(self, *channels, bias=True, norm=None, act=None, drop=0., final_bias=True, final_norm=None, final_act=None, final_drop=0., **kwargs):
        modules = []

        for i in range(1, len(channels) - 1):
            modules.append(self.layer(channels[i - 1], channels[i], bias, norm, act, drop, **kwargs))

        modules.append(self.layer(channels[-2%len(channels)], channels[1], final_bias, final_norm, final_act, final_drop, **kwargs))
        super().__init__(*modules)

    @staticmethod
    def layer(in_channels, out_channels=None, bias=True, norm=None, act=None, drop=0., **kwargs):
        if out_channels is None:
            out_channels = in_channels

        layer_kwargs = dict(tuple(locals().items())[2:-1], **kwargs)
        modules = [nn.Linear(in_channels, out_channels, bias)]

        if norm is not None:
            norm_kwargs = get_kwargs(norm := NORMS[norm_layer], **layer_kwargs)
            modules.append(norm(out_channels, **norm_kwargs))

        if act is not None:
            act_kwargs = get_kwargs(act := ACTS[act_layer], **layer_kwargs)
            modules.append(act(**act_kwargs))

        if drop > 0.:
            modules.append(nn.Dropout(drop))

        module = nn.Sequential(*modules)

        return module

class RNN(MLP):
    @attrmethod
    def __init__(self, channels, bias=True, norm=None, act='tanh', drop=0., seed=None, init_zero=True, **kwargs):
        super().__init__(channels, final_bias=bias, final_norm=norm, final_act=act, final_drop=drop, **kwargs)

        self._state = torch_random_state(seed)

        if init_zero:
            self._init = torch.zeros(1, channels)
        else:
            self._init = torch.rand(1, channels, generator=self._state)

    def forward(self, X=None, n_layers=2):
        if X is None:
            X = self._init

        for i in range(1, n_layers):
            Z = torch.cat((X, super().forward(X[i - 1:i])))

        return Z

class Encoder(nn.Module):
    @attrmethod
    def __init__(self, *channels, bias=True, norm='batch', act='relu', drop=.5, seed=None, **kwargs):
        super().__init__()

        self._state = torch_random_state(seed)
        self._channels = channels if len(channels) > 2 else (channels[0], (channels[0] + channels[-1])//2, channels[-1])
        self._q_net = MLP(*self._channels[:-1], norm=norm, act=act, drop=drop, final_norm=norm, final_act=act, final_drop=drop, **kwargs)
        self._m_mlp = MLP(*self._channels[-2:], final_bias=bias, final_norm=norm, **kwargs)
        self._s_mlp = MLP(*self._channels[-2:], final_bias=bias, **kwargs)

    def forward(self, X, return_kld=False):
        Q = self._q_net(X)
        M, S_log = self._m_mlp(Q), self._s_mlp(Q)
        S_exp = (.5*S_log).exp()
        Z = M + S_exp*torch.randn(M.shape, generator=self._state)

        if return_kld:
            kld = (M**2 + S_exp**2 - S_log - .5).sum()

            return Z, kld
        return Z

class VAE(BaseEstimator, TransformerMixin, nn.Module):
    @attrmethod
    def __init__(self, *channels, bias=True, norm='batch', act='relu', drop=.5, kld_scale=.1, optim='adam', desc='VAE', seed=None, **kwargs):
        super().__init__(**kwargs)

        self._channels = channels if len(channels) > 0 else (64, 32)
        self._n_steps = 100
        self._step_n = 0

    def __call__(self, X, eval=True, **kwargs):
        local_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        transform_kwargs = get_kwargs(self._transform, **call_kwargs)
        Z = self._transform(**transform_kwargs)

        return Z

    def _step(self):
        loss = 0.

        for x in self._loader:
            z, kld = self._encoder(x, return_kld=True)
            x_ = self._decoder(z)
            x_loss = (x_ - x).square().sum().sqrt() + kld*self.kld_scale
            x_loss.backward()
            loss += x_loss.item()

        self._optim.step()
        self._optim.zero_grad()

        return loss

    def _transform(self, X, eval=True):
        if eval:
            self.eval()

        return self._encoder(X).detach()

    def _display(self, label='score'):
        desc = self.desc + '  ' if self.desc is not None else ''
        msg = f'{desc}step: {self._step_n}'

        for k, v in logs:
            if len(v) > 0:
                msg += f'  {k[:-5] if len(k) > 5 else label}: {v[-1]}'

        print(msg)

    def _setup(self, X, n_steps=None, learn_rate=1e-2, batch_size=-1, shuffle=True):
        if n_steps is not None:
            self._n_steps = n_steps
        
        if batch_size < 0:
            batch_size = X.shape[0]//-batch_size

        self._loader = DataLoader(X, batch_size, shuffle)
        self._encoder = Encoder(X.shape[1], *self._channels, bias=self.bias, norm=self.norm, act=self.act, drop=self.drop)
        self._decoder = MLP(*self._channels[::-1], X.shape[1], bias=self.bias, norm=self.norm, act=self.act, drop=self.drop)
        self._optim = OPTIMS[self.optim](self.parameters(), lr=learn_rate)

        self.logs_ = {k : v for k, v in self.__dict__.items() if k[-4:] == 'log_'}
        self.train()

    @checkmethod
    @buildmethod('_setup', '_build')
    def fit(self, X, n_steps=None, learn_rate=1e-2, batch_size=None, shuffle=True, verbosity=1, display_rate=10, **kwargs):
        local_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        step_kwargs, display_kwargs = get_kwargs(self._step, self._display, **local_kwargs)
        self.log_ = []

        for self._step_n in tqdm(range(self._n_steps), self.desc) if verbosity == 1 else range(self._n_steps):
            self.log_.append(self._step(**step_kwargs))

            if verbosity == 2 and self._step_n%display_rate == 0:
                self._display(**display_kwargs)

        return self

    def transform(self, X, eval=True, **kwargs):
        local_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        transform_kwargs = get_kwargs(self._transform, **local_kwargs)
        Z = self._transform(**transform_kwargs)

        return Z
