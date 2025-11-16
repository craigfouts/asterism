'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ._core import buildmethod, checkmethod
from ..utils import get_kwargs
from ..utils.nets import OPTIM, Encoder, MLP

class VAE(BaseEstimator, TransformerMixin, nn.Module):
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
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate)
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

