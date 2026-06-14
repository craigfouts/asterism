'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from abc import abstractmethod, ABCMeta
from sklearn.base import BaseEstimator, ClusterMixin
from tqdm import tqdm
from ..utils import get_kwargs, pad, random_state, relabel
from ..utils.sugar import attrmethod, buildmethod, checkmethod

__all__ = [
    'Asterism'
]

class Asterism(ClusterMixin, BaseEstimator, metaclass=ABCMeta):
    @attrmethod
    def __init__(self, desc=None, seed=None, *, check=True, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparce=False, ensure_all_finite=True, **kwargs):
        super().__init__(**kwargs)

        self._state = None
        self._n_steps = 200
        self._step_n = 0

    @buildmethod('_setup')
    def __call__(self, X, y=None, **kwargs):
        local_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        predict_kwargs = get_kwargs(self._predict, **local_kwargs)
        labels = relabel(self._predict(**predict_kwargs), y)
        
        return labels

    @abstractmethod
    def _step(self):
        pass

    @abstractmethod
    def _predict(self):
        pass

    def _display(self, label='score'):
        desc = self.desc + ':  ' if self.desc is not None else ''
        msg = f'{desc}step={self._step_n}  {label}={self.log_[-1]}'

        for k, v in self.logs_.items():
            if len(v) > 0:
                msg += f'  {k[:-5]}: {v[-1]}'

        print(msg)

    def _setup(self, X, locs=None, n_steps=None):
        self.logs_ = {k: v for k, v in self.__dict__.items() if k.endswith('log_')}

        if self._state is None:
            self._state = random_state(self.seed, isinstance(X, torch.Tensor))

        if locs is not None:
            self._locs = pad(locs, ((n := 3 - locs.shape[1])*(n > 0), 0))

        if n_steps is not None:
            self._n_steps = n_steps

    @checkmethod
    @buildmethod('_setup', '_build')
    def fit(self, X, y=None, locs=None, n_steps=None, verbosity=1, display_rate=10, **kwargs):
        local_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        step_kwargs, predict_kwargs, display_kwargs = get_kwargs(self._step, self._predict, self._display, **local_kwargs)
        self.log_ = []

        for self._step_n in tqdm(range(self._n_steps), self.desc) if verbosity == 1 else range(self._n_steps):
            self.log_.append(self._step(**step_kwargs))

            if verbosity == 2 and self._step_n%display_rate == 0:
                self._display(**display_kwargs)

        self.labels_ = relabel(self._predict(**predict_kwargs), y)

        return self
