"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import torch
from torch import nn
from base import buildmethod, HotTopic
from nets import OPTIM, MLP
from utils import shuffle

class Encoder(nn.Module):
    def __init__(self, in_channels, wc_channels=(128, 128), bc_channels=(128, 128), lp_channels=(128, 128), act_layer='prelu'):
        super().__init__()

        self.in_channels = in_channels
        self.wc_channels = (wc_channels,) if isinstance(wc_channels, int) else wc_channels
        self.bc_channels = (bc_channels,) if isinstance(bc_channels, int) else bc_channels
        self.lp_channels = (lp_channels,) if isinstance(lp_channels, int) else lp_channels
        self.lp_channels += (1,)*(self.lp_channels[-1] != 1)
        self.act_layer = act_layer

        self._wc_model = MLP(self.in_channels, *self.wc_channels, act_layer=self.act_layer)
        self._us_model = MLP(self.in_channels, *self.wc_channels, act_layer=self.act_layer)
        self._bc_model = MLP(self.wc_channels[-1], *self.bc_channels, act_layer=self.act_layer)
        self._lp_model = MLP(self.wc_channels[-1] + self.bc_channels[-1], *self.lp_channels, act_layer=self.act_layer, final_bias=False)

    def _build(self, X):
        self._batch_size = X.shape[0] if X.ndim > 2 else 1
        self._n_samples, self._n_topics = X.shape[-2], 1
        self._wc, self._us = self._wc_model(X), self._us_model(X)
        self._WC = torch.zeros((self._batch_size, 1, self.wc_channels[-1]))
        self._WC[:, 0], self._US = self._wc[:, 0], self._us[:, 2:].sum(1)

        return self
    
    def _update(self, idx, topics):
        n_topics = topics[:idx].unique().shape[0]

        if n_topics == self._n_topics:
            self._WC[:, topics[idx - 1]] += self._wc[:, idx - 1]
        else:
            self._WC = torch.cat((self._WC, self._wc[:, idx - 1].unsqueeze(1)), 1)

        if idx == self._n_samples - 1:
            self._US = torch.zeros((self._batch_size, self.wc_channels[-1]))
        else:
            self._US -= self._us[:, idx]

        self._n_topics = n_topics

        return n_topics
    
    def _logprobs(self, idx):
        US_k = self._US.repeat(self._n_topics, 1, 1)
        WC_k = self._WC.repeat(self._n_topics, 1, 1, 1)
        topic_range = torch.arange(self._n_topics)
        WC_k[topic_range, :, topic_range] += self._wc[:, idx]
        WC_K = torch.cat((self._WC, self._wc[:, idx].unsqueeze(1)), 1)
        BC_k, BC_K = self._bc_model(WC_k).sum(2), self._bc_model(WC_K).sum(1)
        logprobs = torch.zeros((self._batch_size, self._n_topics + 1))
        logprobs[:, :-1] = self._lp_model(torch.cat((US_k, BC_k), -1))[..., 0].T
        logprobs[:, -1] = self._lp_model(torch.cat((self._US, BC_K), 1)).squeeze()
        m, _ = logprobs.max(1, keepdim=True)
        logprobs = logprobs - m - (logprobs - m).exp().sum(1, keepdim=True).log()

        return logprobs
    
    @buildmethod
    def evaluate(self, X, y):
        nll = 0

        for i in range(2, self._n_samples):
            self._update(i, y)
            logprobs = self._logprobs(i)
            nll -= logprobs[:, y[i]].mean()

        return nll
    
    @buildmethod
    def forward(self, X):
        z = torch.zeros(X.shape[-2], dtype=torch.int32)

        for i in range(2, self._n_samples):
            self._update(i, z)
            probs = self._logprobs(i).exp()
            z[i] = torch.mode(torch.multinomial(probs, 1).squeeze()).values.item()

        return z

class NCP(HotTopic, nn.Module):
    def __init__(self, wc_channels=(128, 128), bc_channels=(512, 512), lp_channels=(128, 128), act_layer='prelu', optim='adam', desc='NCP'):
        super().__init__(desc, check=False)

        self.wc_channels = wc_channels
        self.bc_channels = bc_channels
        self.lp_channels = lp_channels
        self.act_layer = act_layer
        self.optim = optim

    def _build(self, X, learning_rate=1e-4, weight_decay=1e-2, batch_size=16):
        self._encoder = Encoder(X.shape[-1], self.wc_channels, self.bc_channels, self.lp_channels, self.act_layer)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._batch_size = X.shape[0] if X.ndim > 2 and X.shape[0] > batch_size else batch_size

        return self
    
    def _step(self, X, y, n_perms=6, batch_size=16, n_samples=64):
        mask, nll = torch.randperm(X.shape[0])[:batch_size], 0

        for _ in range(n_perms):
            data, labels = shuffle(X[mask], y, sort=True, cut=n_samples)
            perm_nll = self._encoder.evaluate(data, labels)
            perm_nll.backward()
            nll += perm_nll.item()

        self._optim.step()
        self._optim.zero_grad()

        return nll
    
    def _predict(self, X):
        if X.ndim < 3 or X.shape[0] == 1:
            X = X.repeat(self._batch_size, 1, 1)

        topics = self._encoder(X)

        return topics
    
    def forward(self, X):
        topics = self._predict(X)

        return topics
