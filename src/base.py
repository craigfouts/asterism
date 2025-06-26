import numpy as np
import torch
from abc import ABC, abstractmethod
from functools import singledispatch, wraps
from inspect import getcallargs, signature
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from tqdm import tqdm

@singledispatch
def check(data, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    if isinstance(data, (tuple, list)):
        data = np.array(data)

    data = check_array(data, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse, ensure_all_finite=ensure_all_finite, ensure_min_features=ensure_min_features)

    if np.iscomplex(data.any()) and not accept_complex:
        raise ValueError('Complex data not supported.')

    return data

@check.register(torch.Tensor)
def _(data, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    if isinstance(data, (tuple, list)):
        data = torch.Tensor(data)
    
    data = torch.tensor(check_array(data, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse, ensure_all_finite=ensure_all_finite, ensure_min_features=ensure_min_features))

    if torch.is_complex(data) and not accept_complex:
        raise ValueError('Complex data not supported.')

    return data

def checkmethod(method, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
    @wraps(method)
    def wrap(self, data, *args, **kwargs):
        data = check(data, 
            self.ensure_min_features if hasattr(self, 'ensure_min_features') else ensure_min_features, 
            self.accept_complex if hasattr(self, 'accept_complex') else accept_complex,
            self.accept_sparse if hasattr(self, 'accept_sparse') else accept_sparse, 
            self.accept_large_sparse if hasattr(self, 'accept_large_sparse') else accept_large_sparse, 
            self.ensure_all_finite if hasattr(self, 'ensure_all_finite') else ensure_all_finite)

        return method(self, data, *args, **kwargs)
    return wrap

def buildmethod(method):
    @wraps(method)
    def wrap(self, *args, **kwargs):
        if hasattr(self, '_build'):
            method_kwargs = dict(getcallargs(method, self, *args), **kwargs)
            build_params = signature(self._build).parameters.keys()
            build_kwargs = {key:val for key, val in method_kwargs.items() if key in build_params}
            self._build(**build_kwargs)

        return method(self, *args, **kwargs)
    return wrap

class HotTopic(ABC, ClusterMixin, BaseEstimator):
    def __init__(self, desc=None, *, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
        super().__init__()

        self.desc = desc
        self.ensure_min_features = ensure_min_features
        self.accept_complex = accept_complex
        self.accept_sparse = accept_sparse
        self.accept_large_sparse = accept_large_sparse
        self.ensure_all_finite = ensure_all_finite

        self._step_n = 0

    @abstractmethod
    def _step(self):
        pass

    @abstractmethod
    def _predict(self):
        pass

    def _display(self):
        desc = self.desc + '  ' if self.desc is not None else ''
        print(f'{desc}step: {self._step_n}  score: {self.log_[-1]}')

    @checkmethod
    @buildmethod
    def fit(self, X, y=None, n_steps=100, verbosity=1, desc=None, rate=10, **kwargs):
        fit_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        step_kwargs = {key:fit_kwargs[key] for key in signature(self._step).parameters.keys()}
        display_kwargs = {key:fit_kwargs[key] for key in signature(self._display).parameters.keys()}
        predict_kwargs = {key:fit_kwargs[key] for key in signature(self._predict).parameters.keys()}
        self.log_ = []

        for self._step_n in tqdm(range(n_steps), desc) if verbosity == 1 else range(n_steps):
            self.log_.append(self._step(**step_kwargs))

            if verbosity == 2 and self._step_n%rate == 0:
                self._display(**display_kwargs)

        self.labels_ = self._predict(**predict_kwargs)

        return self
    
    def fit_predict(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        
        return self.labels_
