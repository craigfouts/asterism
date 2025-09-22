'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_cluster import knn
from torch_geometric.nn.conv import SimpleConv
from ...base import Asterism
from ...utils.nets import OPTIM, Encoder, MLP, RNN

class ATLAS(Asterism, nn.Module):
    def __init__(self, min_topics=1, *, channels=(128, 32), doc_size=8, topic_rate=.2, kld_scale=.1, optim='adam', desc='ATLAS', seed=None):
        super().__init__(desc, seed)

        self.min_topics = min_topics
        self.channels = channels
        self.doc_size = doc_size
        self.topic_rate = topic_rate
        self.kld_scale = kld_scale
        self.optim = optim

        self.n_topics_ = min_topics
        self._channels = (channels,) if isinstance(channels, int) else channels
        self._n_steps = 1000
        self.topic_log_ = []

    def _build(self, X, locs, learning_rate=1e-2, batch_size=128, shuffle=True):
        if batch_size == -1:
            batch_size = X.shape[0]

        self._conv, edges = SimpleConv(), knn(locs, locs, self.doc_size)
        self._loader = DataLoader(self._conv(X, edges), batch_size, shuffle) 
        self._encoder = Encoder(X.shape[1], *self._channels, act_layer='prelu')
        self._dt_rnn = RNN(self._channels[-1], bias=False, act_layer='prelu')
        self._tw_rnn = RNN(self._channels[-1], bias=False, act_layer='prelu')
        self._decoder = MLP(self._channels[-1], X.shape[1], final_bias=False)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate)
        self.train()

        return self
    
    def _generate(self, Z=None, n_topics=None):
        if n_topics is None:
            n_topics = self.n_topics_

        weights = self._decoder(self._tw_rnn(n_layers=n_topics + 1))

        if Z is not None:
            X = F.softmax(Z@self._dt_rnn(n_layers=n_topics + 1).T, -1)@weights

            return X, weights
        return weights
    
    def _evaluate(self, X):
        Z, kld = self._encoder(X, return_kld=True)
        X_k, _ = self._generate(Z, self.n_topics_ - 1)
        X_K, weights = self._generate(Z)
        loss_k = (X_k - X).square().sum(-1)/X.shape[0]
        loss_K = (X_K - X).square().sum(-1)/X.shape[0]

        if self.n_topics_ > (X@weights.T).argmax(-1).unique().shape[0]:
            self.n_topics_ -= 1
        elif (loss_k - loss_K).sum()/loss_K.sum() > self.topic_rate:
            self.n_topics_ += 1

        loss = loss_K.sum().sqrt() + self.kld_scale*kld

        return loss
    
    def _step(self):
        loss = 0.

        for X in self._loader:
            X_loss = self._evaluate(X)
            X_loss.backward()
            loss += X_loss.item()

        self._optim.step()
        self._optim.zero_grad()
        self.topic_log_.append(self.n_topics_)

        return loss
    
    def _predict(self, X, locs, eval=True):
        if eval:
            self.eval()

        edges = knn(locs, locs, self.doc_size)
        X_, weights = self._conv(X, edges), self._generate()
        topics = (X_@weights.T).argmax(-1).detach()

        return topics
