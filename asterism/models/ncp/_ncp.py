'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from torch import nn
from torch.nn import functional as F
from ...base import buildmethod, Asterism
from ...utils import log_norm, shuffle
from ...utils.nets import OPTIM, MLP

def split(X, s, y):
    n_datasets, n_samples = s[:, 0].unique().shape[0] - 1, (s[:, 0] == 0).sum()
    split = X.shape[0] - X[s[:, 0] == n_datasets].shape[0]
    X_train, y_train = X[:split].view(n_datasets, split//n_datasets, X.shape[-1]), y[:n_samples]
    X_test = (X[split:] - F.pad(n_datasets*torch.ones((n_samples, 1)), (0, X.shape[-1] - 1)))[None]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test

class Encoder(nn.Module):
    def __init__(self, in_channels, *, wc_channels=(128, 128), bc_channels=(512, 512), lp_channels=(128, 128)):
        super().__init__()

        self.in_channels = in_channels
        self.wc_channels = wc_channels
        self.bc_channels = bc_channels
        self.lp_channels = lp_channels

        if lp_channels[-1] != 1:
            self.lp_channels += (1,)

        self._wc_mlp = MLP(in_channels, *wc_channels, act_layer='prelu')
        self._us_mlp = MLP(in_channels, *wc_channels, act_layer='prelu')
        self._bc_mlp = MLP(wc_channels[-1], *bc_channels, act_layer='prelu')
        self._lp_mlp = MLP(wc_channels[-1] + bc_channels[-1], *self.lp_channels, act_layer='prelu', final_bias=False)

    def _build(self, X):
        self._batch_size = X.shape[0] if X.ndim > 2 else 1
        self._n_samples, self.n_topics_ = X.shape[-2], 1
        self._topic_range = torch.arange(self.n_topics_)
        self._wc, self._us = self._wc_mlp(X), self._us_mlp(X)
        self._WC = torch.zeros((self._batch_size, 1, self.wc_channels[-1]))
        self._WC[:, 0], self._US = self._wc[:, 0], self._us[:, 2:].sum(1)

        return self
    
    def _update(self, idx, topics):
        n_topics = topics[:idx].unique().shape[0]

        if n_topics == self.n_topics_:
            self._WC[:, topics[idx - 1]] += self._wc[:, idx - 1]
        else:
            self._WC = torch.cat((self._WC, self._wc[:, idx - 1].unsqueeze(1)), 1)

        if idx == self._n_samples - 1:
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
        log_probs = log_norm(log_probs)

        return log_probs
    
    @buildmethod
    def evaluate(self, X, y):
        nll = 0

        for i in range(1, self._n_samples):
            self._update(i, y)
            log_probs = self._generate(i)
            nll -= log_probs[:, y[i]].mean()

        return nll
    
    @buildmethod
    def forward(self, X):
        z = torch.zeros(X.shape[-2], dtype=torch.int32)

        for i in range(1, self._n_samples):
            self._update(i, z)
            probs = self._generate(i).exp()
            z[i] = probs.multinomial(1).squeeze().mode().values.item()

        return z

class NCP(Asterism, nn.Module):
    def __init__(self, *, wc_channels=(128, 128), bc_channels=(512, 512), lp_channels=(128, 128), optim='adam', desc='NCP', seed=None):
        super().__init__(desc, seed, check=False)

        self.wc_channels = wc_channels
        self.bc_channels = bc_channels
        self.lp_channels = lp_channels
        self.optim = optim

        self._n_steps = 200

    def _build(self, X, learning_rate=1e-4, weight_decay=1e-2, batch_size=16):
        self._batch_size = X.shape[0] if X.ndim > 2 and X.shape[0] > 1 else batch_size
        self._encoder = Encoder(X.shape[-1], wc_channels=self.wc_channels, bc_channels=self.bc_channels, lp_channels=self.lp_channels)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train()

        return self
    
    def _step(self, X, y, n_perms=6, n_samples=64):
        mask, nll = torch.randperm(X.shape[0])[:self._batch_size], 0

        for _ in range(n_perms):
            data, labels = shuffle(X[mask], y, sort=True, cut=n_samples)
            perm_nll = self._encoder.evaluate(data, labels)
            perm_nll.backward()
            nll += perm_nll.item()

        self._optim.step()
        self._optim.zero_grad()

        return nll
    
    def _predict(self, X, eval=True):
        if eval:
            self.eval()

        if X.ndim < 3 or X.shape[0] == 1:
            X = X.repeat(self._batch_size, 1, 1)

        topics = self._encoder(X)

        return topics
