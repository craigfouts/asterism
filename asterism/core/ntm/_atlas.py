'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn.conv import SimpleConv
from ...base import Asterism
from ...nets import OPTIMS, Encoder, MLP, RNN
from ...utils import knn2D
from ...utils.sugar import attrmethod

__all__ = [
    'ATLAS'  # Line 20
]

class ATLAS(Asterism, nn.Module):
    @attrmethod
    def __init__(self, min_topics=1, *, channels=(128, 32), doc_size=16, topic_rate=8., kld_scale=.1, optim='adam', desc='ATLAS', seed=None):
        super().__init__(desc, seed)

        self._channels = (channels,) if isinstance(channels, int) else channels
        self._n_steps = 1000
        self.n_topics_ = min_topics
        self.topic_log_ = []

    def _build(self, x, locs, learn_rate=1e-2, batch_size=32, shuffle=True):
        if batch_size < 0:
            batch_size = x.shape[0]//-batch_size

        self._data = SimpleConv(aggr='mean')(x, knn2D(locs, self.doc_size))
        in_channels, out_channels = self._data.shape[-1], self._channels[-1]
        self._loader = DataLoader(self._data, batch_size, shuffle)
        self._encoder = Encoder(in_channels, *self._channels, act='prelu', seed=self._state)
        self._dt_net = RNN(out_channels, bias=False, act='prelu', seed=self._state)
        self._tw_net = RNN(out_channels, bias=False, act='prelu', seed=self._state)
        self._decoder = MLP(out_channels, in_channels, final_bias=False)
        self._optim = OPTIMS[self.optim](self.parameters(), lr=learn_rate)
        self.train()

        return self
    
    def _generate(self, z=None, n_topics=-1):
        if n_topics == -1:
            n_topics = self.n_topics_

        w = self._decoder(self._tw_net(n_layers=n_topics))

        if z is not None:
            x = F.softmax(z@self._dt_net(n_layers=n_topics).T, -1)@w

            return x, w
        return w
    
    def _evaluate(self, x):
        z, kld = self._encoder(x, return_kld=True)
        x_k, w = self._generate(z)
        x_K, _ = self._generate(z, self.n_topics_ + 1)
        n_topics = (x@w.T).argmax(-1).unique().shape[0]
        loss_k = (x_k - x).square().sum(-1)/(n_pts := x.shape[0])
        loss_K = (x_K - x).square().sum(-1)/n_pts
        loss = loss_K.sum() + self.kld_scale*kld
        rate = (loss_k - loss_K).sum()/loss_K.sum()

        if n_topics < self.n_topics_:
            self.n_topics_ -= 1
        elif self.topic_rate > 0. and rate > 1./self.topic_rate:
            self.n_topics_ += 1

        return loss
    
    def _step(self):
        loss = 0.

        for x in self._loader:
            (x_loss := self._evaluate(x)).backward()
            loss += x_loss.item()

        self._optim.step()
        self._optim.zero_grad()
        self.topic_log_.append(self.n_topics_)

        return loss
    
    def _predict(self, train=False):
        if train:
            self.train()
        else:
            self.eval()

        z = self._data@self._generate().T
        topics = z.argmax(-1).detach()

        return topics

    def fit(self, x, locs, y=None, *args, **kwargs):
        super().fit(x, y, locs, *args, **kwargs)

        return self

    def fit_predict(self, x, locs, y=None, *args, **kwargs):
        super().fit(x, y, locs, *args, **kwargs)

        return self.labels_
