'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from abc import abstractmethod, ABCMeta
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.multiclass import type_of_target
from tqdm import tqdm
from ..utils import get_kwargs, pad, random_state, relabel, to_tensor
from ..utils.sugar import attrmethod, buildmethod, checkmethod

__all__ = [
    'Asterism'  # Line 18
]

class Asterism(ClusterMixin, BaseEstimator, metaclass=ABCMeta):
    @attrmethod
    def __init__(self, desc=None, seed=None, *, check=True, ensure_min_features=1, accept_complex=False, accept_sparse=False, accept_large_sparse=False, ensure_all_finite=True, torch_model=False):
        super().__init__()

        self._n_steps = 200
        self._step_n = 0

    def __call__(self, x, y=None, **kwargs):
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

    def __check(self, locs=None, n_steps=None):
        if not hasattr(self, '_state'):
            self._state = random_state(self.seed, self.torch_model)

        if locs is not None:
            self._locs = pad(locs, ((n := 3 - locs.shape[1])*(n > 0), 0))

        if n_steps is not None:
            self._n_steps = n_steps

    @buildmethod('_Asterism__check')
    def __setup(self, x, y=None, locs=None, n_steps=None):
        self._tensor_io = isinstance(x, torch.Tensor)
        self.n_features_in_, self.log_ = x.shape[-1], []
        self.logs_ = {k: v for k, v in self.__dict__.items() if k.endswith('log_')}

        if y is not None:
            type_of_target(y, raise_unknown=True)

        if self.torch_model:
            x, y = to_tensor(x, y)

        return {'x': x, 'y': y}

    @checkmethod
    @buildmethod('_Asterism__setup', '_build')
    def fit(self, x, y=None, locs=None, n_steps=None, verbosity=1, display_rate=10, **kwargs):
        local_kwargs = dict(tuple(locals().items())[:-1], **kwargs)
        step_kwargs, predict_kwargs, display_kwargs = get_kwargs(self._step, self._predict, self._display, **local_kwargs)

        for self._step_n in tqdm(range(self._n_steps), self.desc) if verbosity == 1 else range(self._n_steps):
            self.log_.append(self._step(**step_kwargs))

            if verbosity == 2 and self._step_n%display_rate == 0:
                self._display(**display_kwargs)

        self.labels_ = relabel(self._predict(**predict_kwargs), y)

        if self._tensor_io:
            self.labels_ = to_tensor(self.labels_)
        elif self.torch_model:
            self.labels_ = self.labels_.numpy()

        return self
