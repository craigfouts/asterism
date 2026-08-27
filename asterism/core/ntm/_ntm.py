'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn
from torch.utils.data import DataLoader
from ...base import Asterism
from ...nets import OPTIMS, Encoder, MLP
from ...utils.sugar import attrmethod

__all__ = [
    'NTM'  # Line 17
]

class NTM(Asterism, nn.Module):
    @attrmethod
    def __init__(self, n_topics, *, channels=(128, 32), kld_scale=.1, mode='softmax', optim='adam', desc='NTM', seed=None):
        super().__init__(desc, seed)

        if mode.lower() not in ('softmax', 'dirichlet'):
            raise ValueError(f'Mode `{mode}` not supported.')

        self._channels = (channels,) if isinstance(channels, int) else channels
        self._n_steps = 2000
    
    def _build(self, x, learn_rate=1e-2, batch_size=32, shuffle=True):
        if batch_size < 0:
            batch_size = x.shape[0]//-batch_size
            
        self._data, in_channels = x, x.shape[-1]
        out_channels = self.n_topics - (self.mode == 'dirichlet')
        self._loader = DataLoader(self._data, batch_size, shuffle, generator=self._state)
        self._encoder = Encoder(in_channels, *self._channels, seed=self._state)
        self._dt_net = MLP(self._channels[-1], out_channels, final_act=self.mode, dim=-1)
        self._decoder = MLP(self.n_topics, in_channels, final_bias=False)
        self._optim = OPTIMS[self.optim](self.parameters(), lr=learn_rate)
        self.train()

        return self
    
    def _evaluate(self, x):
        z, kld = self._encoder(x, return_kld=True)
        x_ = self._decoder(self._dt_net(z))
        x_loss = (x_ - x).square().sum()/x.shape[0]
        loss = x_loss + self.kld_scale*kld

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
            self.train()
        else:
            self.eval()

        z = self._data@self._decoder[0][0].weight
        topics = z.argmax(-1).detach()

        return topics
