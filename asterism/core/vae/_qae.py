'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from ...base import Asterism
from ...nets import OPTIMS, MLP, Encoder
from ...utils import fps
from ...utils.sugar import attrmethod

__all__ = [
    'VQAE'  # Line 19
]
        
class VQAE(Asterism, nn.Module):
    @attrmethod
    def __init__(self, n_topics, *, channels=(128, 32), optim='adam', desc='VQAE', seed=None):
        super().__init__(desc, seed)

        self._channels = (channels,) if isinstance(channels, int) else channels
        self._n_steps = 200
        
    def _build(self, x, learn_rate=1e-3, batch_size=32, shuffle=True):
        if batch_size < 0:
            batch_size = x.shape[0]//-batch_size

        self._data, in_channels = x, x.shape[-1]
        self._loader = DataLoader(self._data, batch_size, shuffle, generator=self._state)
        self._encoder = MLP(in_channels, *self._channels, act='relu')
        self._decoder = MLP(*self._channels[::-1], in_channels, act='relu')
        codebook = fps(self._encoder(self._data).detach(), self.n_topics)
        self._codebook = nn.Parameter(codebook, requires_grad=True)
        self._optim = OPTIMS[self.optim](self.parameters(), lr=learn_rate)
        self.train()

        return self
    
    def _quantize(self, z, z_grad=False, e_grad=False, return_loss=False):
        z_ = z if z_grad else z.detach()
        e_ = self._codebook if e_grad else self._codebook.detach()
        prox = (z_[:, None] - e_[None]).square().sum(-1)
        topics = prox.argmin(-1)
        
        if return_loss:
            loss = prox[torch.arange(topics.shape[0]), topics].sum()

            return topics, loss
        return topics
    
    def _evaluate(self, x):
        z = self._encoder(x)
        topics, z_loss = self._quantize(z, z_grad=True, return_loss=True)
        _, e_loss = self._quantize(z, e_grad=True, return_loss=True)
        x_ = self._decoder(self._codebook[topics])
        loss = z_loss + e_loss + (x_ - x).square().sum()

        return loss
    
    def _step(self):
        loss = 0.

        for x in self._loader:
            (x_loss := self._evaluate(x)).backward()
            loss += x_loss.item()

        self._optim.step()
        self._optim.zero_grad()

        return loss
    
    def _predict(self, train=False):
        if train:
            self.train
        else:
            self.eval()

        topics = self._quantize(self._encoder(self._data))

        return topics
