'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from torch import nn
from torch.nn import functional as F
from ...base import Asterism
from ...nets import OPTIMS, MLP
from ...utils import log_normalize, shuffle
from ...utils.sugar import attrmethod, buildmethod

__all__ = [
    'Encoder',  # Line 20
    'NCP'       # Line 99
]

class Encoder(nn.Module):
    @attrmethod
    def __init__(self, in_channels, *, wc_channels=(128, 128), bc_channels=(512, 512), lp_channels=(128, 128)):
        super().__init__()

        if lp_channels[-1] != 1:
            self.lp_channels += (1,)

        self._wc_mlp = MLP(in_channels, *wc_channels, act='prelu')
        self._us_mlp = MLP(in_channels, *wc_channels, act='prelu')
        self._bc_mlp = MLP(wc_channels[-1], *bc_channels, act='prelu')
        self._lp_mlp = MLP(wc_channels[-1] + bc_channels[-1], *self.lp_channels, act='prelu', final_bias=False)

    def _build(self, x):
        if x.ndim > 2:
            self._batch_size = x.shape[0]
        else:
            self._batch_size = 1

        self._n_pts, self.n_topics_ = x.shape[-2], 1
        self._topic_range = torch.arange(self.n_topics_)
        self._wc, self._us = self._wc_mlp(x), self._us_mlp(x)
        self._WC = torch.zeros((self._batch_size, 1, self.wc_channels[-1]))
        self._WC[:, 0], self._US = self._wc[:, 0], self._us[:, 2:].sum(1)

        return self
    
    def _update(self, idx, topics):
        n_topics = topics[:idx].unique().shape[0]

        if n_topics == self.n_topics_:
            self._WC[:, topics[idx - 1]] += self._wc[:, idx - 1]
        else:
            self._WC = torch.cat((self._WC, self._wc[:, idx - 1].unsqueeze(1)), 1)

        if idx == self._n_pts - 1:
            self._US = torch.zeros((self._batch_size, self.wc_channels[-1]))
        else:
            self._US -= self._us[:, idx]

        self.n_topics_, self._topic_range = n_topics, torch.arange(n_topics)

        return n_topics
    
    def _generate(self, idx):
        WC_k = self._WC.repeat(self.n_topics_, 1, 1, 1)
        WC_k[self._topic_range, :, self._topic_range] += self._wc[:, idx]
        WC_K = torch.cat((self._WC, self._wc[:, idx].unsqueeze(1)), 1)
        BC_k, BC_K = self._bc_mlp(WC_k).sum(2), self._bc_mlp(WC_K).sum(1)
        US_k = self._US.repeat(self.n_topics_, 1, 1)
        log_probs = torch.zeros((self._batch_size, self.n_topics_ + 1))
        log_probs[:, :-1] = self._lp_mlp(torch.cat((BC_k, US_k), -1))[..., 0].T
        log_probs[:, -1] = self._lp_mlp(torch.cat((BC_K, self._US), 1)).squeeze()
        log_probs = log_normalize(log_probs)

        return log_probs
    
    @buildmethod
    def evaluate(self, x, y):
        nll = 0

        for i in range(1, self._n_pts):
            self._update(i, y)
            log_probs = self._generate(i)
            nll -= log_probs[:, y[i]].mean()

        return nll
    
    @buildmethod
    def forward(self, x):
        z = torch.zeros(x.shape[-2], dtype=torch.int32)

        for i in range(1, self._n_pts):
            self._update(i, z)
            probs = self._generate(i).exp()
            z[i] = probs.multinomial(1).squeeze().mode().values.item()

        return z

class NCP(Asterism, nn.Module):
    @attrmethod
    def __init__(self, *, wc_channels=(128, 128), bc_channels=(512, 512), lp_channels=(128, 128), optim='adam', desc='NCP', seed=None):
        super().__init__(desc, seed, check=False)

        self._n_steps = 200

    def _build(self, x, learning_rate=1e-4, weight_decay=1e-2, batch_size=16):
        self._n_pts = x.shape[0]

        if x.ndim > 2 and self._n_pts > 1:
            self._batch_size = self._n_pts
        else:
            self._batch_size = batch_size

        self._encoder = Encoder(x.shape[-1], wc_channels=self.wc_channels, bc_channels=self.bc_channels, lp_channels=self.lp_channels)
        self._optim = OPTIMS[self.optim](self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train()

        return self
    
    def _step(self, x, y, n_perms=4, n_pts=64):
        mask = torch.randperm(self._n_pts, generator=self._state)[:self._batch_size]
        nll = 0

        for _ in range(n_perms):
            x_, y_ = shuffle(x[mask], y, sort=True, cut=n_pts)
            (nll_ := self._encoder.evaluate(x_, y_)).backward()
            nll += nll_.item()

        self._optim.step()
        self._optim.zero_grad()

        return nll

    def _predict(self, x, train=False):
        if train:
            self.train()
        else:
            self.eval()

        if x.ndim < 3 or x.shape[0] == 1:
            x = x.repeat(self._batch_size, 1, 1)

        topics = self._encoder(x).detach()

        return topics
