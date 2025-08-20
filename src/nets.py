"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from base import buildmethod, checkmethod

NORM = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
ACT = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus}
OPTIM = {'adam': optim.Adam, 'sgd': optim.SGD}

class MLP(nn.Sequential):
    def __init__(self, *channels, bias=True, norm_layer=None, act_layer=None, dropout=0., final_bias=True, final_norm=None, final_act=None, final_dropout=0.):
        modules = []

        for i in range(1, len(channels) - 1):
            modules.append(self.layer(channels[i - 1], channels[i], bias, norm_layer, act_layer, dropout))

        modules.append(self.layer(channels[-2%len(channels)], channels[-1], final_bias, final_norm, final_act, final_dropout))

        super().__init__(*modules)

    @staticmethod
    def layer(in_channels, out_channels=None, bias=True, norm_layer=None, act_layer=None, dropout=0.):
        if out_channels is None:
            out_channels = in_channels

        module = nn.Sequential(nn.Linear(in_channels, out_channels, bias))

        if norm_layer is not None:
            module.append(NORM[norm_layer](out_channels))

        if act_layer is not None:
            module.append(ACT[act_layer]())

        if dropout > 0.:
            module.append(nn.Dropout(dropout))

        return module
    
class Encoder(nn.Module):
    def __init__(self, *channels, norm_layer='batch', act_layer='relu', dropout=.2, kld_scale=.1):
        super().__init__()

        self.channels = channels if len(channels) > 2 else (channels[0], (channels[0] + channels[-1])//2, channels[-1])
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.dropout = dropout
        self.kld_scale = kld_scale

        self._e_model = MLP(*self.channels[:-1], norm_layer=self.norm_layer, act_layer=self.act_layer, dropout=self.dropout, final_norm=self.norm_layer, final_act=self.act_layer, final_dropout=dropout)
        self._m_model, self._s_model = MLP(*self.channels[-2:]), MLP(*self.channels[-2:])

    def forward(self, X, return_kld=False):
        e = self._e_model(X)
        m, s_log = self._m_model(e), self._s_model(e)
        s_exp = s_log.exp()
        z = m + s_exp*torch.randn_like(s_exp)

        if return_kld:
            kld = self.kld_scale*(m**2 + s_exp**2 - s_log - .5).sum()

            return z, kld
        return z

class VAE(BaseEstimator, TransformerMixin, nn.Module):
    def __init__(self, n_topics=3, *, channels=(64, 32), norm_layer='batch', act_layer='relu', dropout=.2, kld_scale=.1, optim='adam', desc='VAE', random_state=None):
        super().__init__()

        self.n_topics = n_topics
        self.channels = channels
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.dropout = dropout
        self.kld_scale = kld_scale
        self.optim = optim
        self.desc = desc
        self.random_state = random_state

        self._step_n = 0

    def _build(self, X, learning_rate=1e-2, batch_size=32):
        self._n_samples, in_channels = X.shape
        self._loader = DataLoader(X, batch_size, shuffle=True)
        self._encoder = Encoder(in_channels, *self.channels, norm_layer=self.norm_layer, act_layer=self.act_layer, dropout=self.dropout, kld_scale=self.kld_scale)
        self._decoder = MLP(*self.channels[::-1], in_channels, norm_layer=self.norm_layer, act_layer=self.act_layer, dropout=self.dropout)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate)

        return self
    
    def _step(self):
        loss = 0.

        for x in self._loader:
            z, kl = self._encoder(x, return_kld=True)
            x_ = self._decoder(z)
            x_loss = (x_ - x).square().sum().sqrt() + kl
            x_loss.backward()
            loss += x_loss.item()

        self._optim.step()
        self._optim.zero_grad()

        return loss
    
    def _display(self):
        desc = self.desc + '  ' if self.desc is not None else ''
        msg = f'{desc}step: {self._step_n}'

        if hasattr(self, 'log_'):
            msg += f'  score: {self.log_[-1]}'

        print(msg)

    def forward(self, X):
        if hasattr(self, '_encoder'):
            z = self._encoder(X).detach()

            return z
        return X
    
    @checkmethod
    @buildmethod
    def fit(self, X, n_steps=200, learning_rate=1e-2, batch_size=32, verbosity=1, display_rate=10):
        self.log_ = []
        self.train()

        for self._step_n in tqdm(range(n_steps), self.desc) if verbosity == 1 else range(n_steps):
            self.log_.append(self._step())

            if verbosity == 2 and self._step_n%display_rate == 0:
                self._display()

        self.eval()

        return self
    
    def transform(self, X):
        z = self(X)

        return z
    