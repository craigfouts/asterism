"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from functools import singledispatch, wraps
from inspect import getcallargs
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array, check_random_state
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_kwargs, relabel

class Dirichlet(nn.Module):
    def forward(self, x, sigmoid=True):
        if sigmoid:
            x = F.sigmoid(x)

        products = F.pad((1 - x).cumprod(-1), (1, 0), value=1)
        weights = F.pad(x, (0, 1), value=1)*products

        return weights

NORM = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
ACT = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus, 'softmax': nn.Softmax, 'dirichlet': Dirichlet}
OPTIM = {'adam': optim.Adam, 'sgd': optim.SGD}

@singledispatch
def check(X, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    if isinstance(X, (tuple, list)):
        X = np.array(X)

    X = check_array(X, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse, ensure_all_finite=ensure_all_finite, ensure_min_features=ensure_min_features)

    if not accept_complex and np.iscomplex(X).any():
        raise ValueError('Complex data not supported.')

    return X

@check.register(torch.Tensor)
def _(X, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    if isinstance(X, (tuple, list)):
        X = torch.Tensor(X)
    
    X = torch.tensor(check_array(X, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse, ensure_all_finite=ensure_all_finite, ensure_min_features=ensure_min_features))

    if not accept_complex and torch.is_complex(X):
        raise ValueError('Complex data not supported.')

    return X

def checkmethod(method, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    @wraps(method)
    def wrap(self, X, *args, **kwargs):
        if hasattr(self, 'random_state'):
            self.random_state_ = check_random_state(self.random_state)

        if not hasattr(self, 'check') or self.check:
            X = check(X, 
                self.ensure_min_features if hasattr(self, 'ensure_min_features') else ensure_min_features, 
                self.accept_complex if hasattr(self, 'accept_complex') else accept_complex,
                self.accept_sparse if hasattr(self, 'accept_sparse') else accept_sparse, 
                self.accept_large_sparse if hasattr(self, 'accept_large_sparse') else accept_large_sparse, 
                self.ensure_all_finite if hasattr(self, 'ensure_all_finite') else ensure_all_finite)
            
        return method(self, X, *args, **kwargs)
    return wrap

def buildmethod(method):  # TODO: add 'build' parameter
    @wraps(method)
    def wrap(self, *args, **kwargs):
        if hasattr(self, '_build'):
            method_kwargs = dict(getcallargs(method, self, *args), **kwargs)
            build_kwargs = get_kwargs(self._build, **method_kwargs)
            self._build(**build_kwargs)

        return method(self, *args, **kwargs)
    return wrap

class MLP(nn.Sequential):
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

        layer_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        module = nn.Sequential(nn.Linear(in_channels, out_channels, bias))

        if norm_layer is not None:
            norm_kwargs = get_kwargs(NORM[norm_layer], **layer_kwargs)
            module.append(NORM[norm_layer](out_channels, **norm_kwargs))

        if act_layer is not None:
            act_kwargs = get_kwargs(ACT[act_layer], **layer_kwargs)
            module.append(ACT[act_layer](**act_kwargs))

        if dropout > 0.:
            module.append(nn.Dropout(dropout))

        return module
    
class MLP(nn.Sequential):
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

        layer_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        module = nn.Sequential(nn.Linear(in_channels, out_channels, bias))

        if norm_layer is not None:
            norm_kwargs = get_kwargs(NORM[norm_layer], **layer_kwargs)
            module.append(NORM[norm_layer](out_channels, **norm_kwargs))

        if act_layer is not None:
            act_kwargs = get_kwargs(ACT[act_layer], **layer_kwargs)
            module.append(ACT[act_layer](**act_kwargs))

        if dropout > 0.:
            module.append(nn.Dropout(dropout))

        return module
    
class Encoder(nn.Module):
    def __init__(self, *channels, norm_layer='batch', act_layer='relu', dropout=.2, **kwargs):
        super().__init__()

        self.channels = channels if len(channels) > 2 else (channels[0], (channels[0] + channels[-1])//2, channels[-1])
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.dropout = dropout

        self._e_model = MLP(*self.channels[:-1], act_layer=self.act_layer, final_act=self.act_layer, final_dropout=self.dropout, **kwargs)
        self._m_model = MLP(*self.channels[-2:], final_norm=norm_layer, **kwargs)
        self._s_model = MLP(*self.channels[-2:], final_norm=norm_layer, **kwargs)

    def forward(self, X, return_kld=False):
        e = self._e_model(X)
        m, s_log = self._m_model(e), self._s_model(e)
        s_exp = (.5*s_log).exp()
        z = m + s_exp*torch.randn_like(m)

        if return_kld:
            kld = (m**2 + s_exp**2 - s_log - .5).sum()

            return z, kld
        return z
    
class VAE(BaseEstimator, TransformerMixin, nn.Module):
    def __init__(self, channels=(64, 32), *, norm_layer='batch', act_layer='relu', dropout=.2, kld_scale=.1, optim='adam', desc='VAE'):
        super().__init__()

        self.channels = channels
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.dropout = dropout
        self.kld_scale = kld_scale
        self.optim = optim
        self.desc = desc

        self._step_n = 0

    def __call__(self, X, eval=True):
        z = self.transform(X, eval)

        return z

    def _build(self, X, learning_rate=1e-2, batch_size=32, shuffle=True):
        in_channels = X.shape[-1]
        self._loader = DataLoader(X, batch_size, shuffle)
        self._encoder = Encoder(in_channels, *self.channels, norm_layer=self.norm_layer, act_layer=self.act_layer, dropout=self.dropout)
        self._decoder = MLP(*self.channels[::-1], in_channels, norm_layer=self.norm_layer, act_layer=self.act_layer, dropout=self.dropout)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate)
        self.train()

        return self
    
    def _step(self):
        loss = 0.

        for x in self._loader:
            z, kl = self._encoder(x, return_kld=True)
            x_ = self._decoder(z)
            x_loss = (x_ - x).square().sum().sqrt() + self.kld_scale*kl
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

        z = self._encoder(X).detach()

        return z

class HotTopic(ClusterMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self, desc=None, random_state=None, *, check=True, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
        super().__init__()

        self.desc = desc
        self.random_state = random_state
        self.check = check
        self.ensure_min_features = ensure_min_features
        self.accept_complex = accept_complex
        self.accept_sparse = accept_sparse
        self.accept_large_sparse = accept_large_sparse
        self.ensure_all_finite = ensure_all_finite

        self._n_steps = 100
        self._step_n = 0

    def __call__(self, X, y=None, **kwargs):
        call_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        predict_kwargs = get_kwargs(self._predict, **call_kwargs)
        topics = relabel(self._predict(**predict_kwargs), y)

        return topics

    @abstractmethod
    def _step(self):
        pass

    @abstractmethod
    def _predict(self):
        pass

    def _display(self):
        desc = self.desc + '  ' if self.desc is not None else ''
        msg = f'{desc}step: {self._step_n}'

        if hasattr(self, 'log_'):
            msg += f'  score: {self.log_[-1]}'

        print(msg)

    @checkmethod
    @buildmethod
    def fit(self, X, y=None, n_steps=None, verbosity=1, display_rate=10, **kwargs):
        fit_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        step_kwargs, predict_kwargs, display_kwargs = get_kwargs(self._step, self._predict, self._display, **fit_kwargs)
        self.log_ = []

        if n_steps is not None:
            self._n_steps = n_steps

        for self._step_n in tqdm(range(self._n_steps), self.desc) if verbosity == 1 else range(self._n_steps):
            self.log_.append(self._step(**step_kwargs))

            if verbosity == 2 and self._step_n%display_rate == 0:
                self._display(**display_kwargs)

        self.labels_ = relabel(self._predict(**predict_kwargs), y)

        return self
    
    def fit_predict(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        
        return self.labels_
