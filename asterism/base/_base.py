'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from functools import singledispatch, wraps
from inspect import getcallargs
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_random_state
from tqdm import tqdm
from ..utils import get_kwargs, relabel

@singledispatch
def _check(X, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    if isinstance(X, (tuple, list)):
        X = np.array(X)

    X = check_array(X, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse, ensure_all_finite=ensure_all_finite, ensure_min_features=ensure_min_features)

    if not accept_complex and np.iscomplex(X).any():
        raise ValueError('Complex data not supported.')

    return X

@_check.register(torch.Tensor)
def _(X, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    if isinstance(X, (tuple, list)):
        X = torch.Tensor(X)
    
    X = torch.tensor(check_array(X, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse, ensure_all_finite=ensure_all_finite, ensure_min_features=ensure_min_features))

    if not accept_complex and torch.is_complex(X):
        raise ValueError('Complex data not supported.')

    return X

def checkmethod(method, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    @wraps(method)
    def wrapper(self, X, *args, **kwargs):
        self.seed_ = check_random_state(self.seed if hasattr(self, 'seed') else 0)

        if not hasattr(self, 'check') or self.check:
            X = _check(X, 
                self.ensure_min_features if hasattr(self, 'ensure_min_features') else ensure_min_features, 
                self.accept_complex if hasattr(self, 'accept_complex') else accept_complex,
                self.accept_sparse if hasattr(self, 'accept_sparse') else accept_sparse, 
                self.accept_large_sparse if hasattr(self, 'accept_large_sparse') else accept_large_sparse, 
                self.ensure_all_finite if hasattr(self, 'ensure_all_finite') else ensure_all_finite)
            
        return method(self, X, *args, **kwargs)
    return wrapper

@singledispatch
def buildmethod(method):
    def wrap(self, *args, **kwargs):
        if hasattr(self, '_build'):
            method_kwargs = dict(getcallargs(method, self, *args), **kwargs)
            build_kwargs = get_kwargs(self._build, **method_kwargs)
            self._build(**build_kwargs)

        return method(self, *args, **kwargs)
    return wrap

@buildmethod.register(str)
def _(build):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, build):
                builder = getattr(self, build)
                method_kwargs = dict(getcallargs(method, self, *args), **kwargs)
                build_kwargs = get_kwargs(builder, **method_kwargs)
                builder(**build_kwargs)
            
            return method(self, *args, **kwargs)
        return wrapper
    return decorator

class Asterism(ClusterMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self, desc=None, seed=None, *, check=True, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
        super().__init__()

        self.desc = desc
        self.seed = seed
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
