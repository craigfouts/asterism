'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from tqdm import tqdm
from ..utils import get_kwargs, pad, relabel, torch_random_state
from ..utils.sugar import attrmethod, buildmethod, checkmethod

__all__ = [
    'Asterism',
    'AsterismSpatial'
]

class Asterism(ClusterMixin, BaseEstimator, metaclass=ABCMeta):
    @attrmethod
    def __init__(self, desc=None, seed=None, *, torch_state=False, check=True, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True):
        super().__init__()

        self._state = torch_random_state(seed) if torch_state else check_random_state(seed)
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

        if hasattr(self, 'log_') and len(self.log_) > 0:
            msg += f'  score: {self.log_[-1]}'

        print(msg)

    def _setup(self, n_steps=None):
        if n_steps is not None:
            self._n_steps = n_steps

        return self

    @checkmethod
    @buildmethod('_setup', '_build')
    def fit(self, X, y=None, n_steps=None, verbosity=1, display_rate=10, **kwargs):
        fit_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        step_kwargs, predict_kwargs, display_kwargs = get_kwargs(self._step, self._predict, self._display, **fit_kwargs)
        self.log_ = []

        for self._step_n in tqdm(range(self._n_steps), self.desc) if verbosity == 1 else range(self._n_steps):
            self.log_.append(self._step(**step_kwargs))

            if verbosity == 2 and self._step_n%display_rate == 0:
                self._display(**display_kwargs)

        self.labels_ = relabel(self._predict(**predict_kwargs), y)

        return self

class AsterismSpatial(Asterism):
    def _setup(self, locs, n_steps=None):
        super()._setup(n_steps)
        self._locs = pad(locs, ((n := 3 - locs.shape[1])*(n > 0), 0))

        return self

    @checkmethod
    @buildmethod('_setup', '_build')
    def fit(self, X, locs, y=None, n_steps=None, verbosity=1, display_rate=10, **kwargs):
        fit_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        step_kwargs, predict_kwargs, display_kwargs = get_kwargs(self._step, self._predict, self._display, **fit_kwargs)
        self.log_ = []

        for self._step_n in tqdm(range(self._n_steps), self.desc) if verbosity == 1 else range(self._n_steps):
            self.log_.append(self._step(**step_kwargs))

            if verbosity == 2 and self._step_n%display_rate == 0:
                self._display(**display_kwargs)

        self.labels_ = relabel(self._predict(**predict_kwargs), y)

        return self

    def fit_predict(self, X, locs, y=None, n_steps=None, verbosity=1, display_rate=10, **kwargs):
        self.fit(X, locs, y, n_steps, verbosity, display_rate, **kwargs)

        return self.labels_
