'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn
from torch.utils.data import DataLoader
from ...base import Asterism
from ...utils.nets import OPTIM, Encoder, MLP

class NTM(Asterism, nn.Module):
    def __init__(self, max_topics=100, *, channels=(128, 32), kld_scale=.1, mode='softmax', optim='adam', desc='NTM', seed=None):
        super().__init__(desc, seed)

        self.max_topics = max_topics
        self.channels = channels
        self.kld_scale = kld_scale
        self.mode = mode
        self.optim = optim

        if mode not in ('softmax', 'dirichlet'):
            raise ValueError(f'Mode `{mode}` not supported.')

        self._n_steps = 1000
    
    def _build(self, X, learning_rate=1e-2, batch_size=128, shuffle=True):
        out_channels = self.max_topics - (self.mode == 'dirichlet')
        self._loader = DataLoader(X, batch_size, shuffle)
        self._encoder = Encoder(X.shape[1], *self.channels)
        self._g_model = MLP(self.channels[-1], out_channels, final_act=self.mode, dim=-1)
        self._decoder = MLP(self.max_topics, X.shape[1], final_bias=False)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate)
        self.train()

        return self
    
    def _evaluate(self, X):
        z, kld = self._encoder(X, return_kld=True)
        X_ = self._decoder(self._g_model(z))
        loss = (X_ - X).square().sum()/X.shape[0] + self.kld_scale*kld

        return loss
    
    def _step(self):
        loss = 0.

        for x in self._loader:
            x_loss = self._evaluate(x)
            x_loss.backward()
            loss += x_loss.item()

        self._optim.step()
        self._optim.zero_grad()

        return loss
    
    def _predict(self, X, eval=True):
        if eval:
            self.eval()

        topics = (X@self._decoder[0][0].weight.detach()).argmax(-1)

        return topics
