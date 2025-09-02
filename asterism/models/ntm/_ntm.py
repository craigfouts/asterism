'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn
from torch.utils.data import DataLoader
from ...base import Keynote
from ...utils.nets import OPTIM, Encoder, MLP

class NTM(Keynote, nn.Module):
    def __init__(self, max_topics=100, *, channels=(128, 32), kld_scale=.1, mode='softmax', optim='adam', desc='NTM', seed=None):
        super().__init__(desc, seed)

        self.max_topics = max_topics
        self.channels = channels
        self.kld_scale = kld_scale
        self.mode = mode
        self.optim = optim

        if self.mode not in ('softmax', 'dirichlet'):
            raise ValueError(f'Mode "{self.mode}" not supported.')

        self._n_steps = 1000
    
    def _build(self, X, learning_rate=1e-2, batch_size=128, shuffle=True):
        in_channels, self._batch_size = X.shape[-1], batch_size
        out_channels = self.max_topics - (self.mode == 'dirichlet')
        self._loader = DataLoader(X, self._batch_size, shuffle)
        self._encoder = Encoder(in_channels, *self.channels)
        self._g_model = MLP(self.channels[-1], out_channels, final_act=self.mode, dim=-1)
        self._decoder = MLP(self.max_topics, in_channels, final_bias=False)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate)
        self.train()

        return self
    
    def _step(self):
        loss = 0.

        for x in self._loader:
            z, kl = self._encoder(x, return_kld=True)
            x_ = self._decoder(self._g_model(z))
            x_loss = (x_ - x).square().sum().sqrt() + self.kld_scale*kl
            x_loss.backward()
            loss += x_loss.item()/self._batch_size

        self._optim.step()
        self._optim.zero_grad()

        return loss
    
    def _predict(self, X, eval=True):
        if eval:
            self.eval()

        topics = (X@self._decoder[0][0].weight.detach()).argmax(-1)

        return topics
    
    def forward(self, X, eval=True):
        topics = self._predict(X, eval)

        return topics
