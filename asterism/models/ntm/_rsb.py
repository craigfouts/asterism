'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ...base import Asterism
from ...utils.nets import OPTIM, Encoder, MLP, RNN

class RSB(Asterism, nn.Module):
    def __init__(self, min_topics=2, topic_rate=.2, *, channels=(64, 32), kld_scale=.1, optim='adam', desc='RSB', seed=None):
        super().__init__(desc, seed)

        self.min_topics = min_topics
        self.topic_rate = topic_rate
        self.channels = channels
        self.kld_scale = kld_scale
        self.optim = optim

        self.n_topics_ = min_topics
        self._channels = (channels,) if isinstance(channels, int) else channels
        self._n_steps = 1000
        self.topic_log_ = []

    def _build(self, X, learning_rate=1e-2, batch_size=128, shuffle=True):
        self._loader = DataLoader(X, batch_size, shuffle)
        self._encoder = Encoder(X.shape[1], *self.channels, act_layer='prelu')
        self._dt_rnn = RNN(self.channels[-1], bias=False, act_layer='prelu')
        self._tw_rnn = RNN(self.channels[-1], bias=False, act_layer='prelu')
        self._decoder = MLP(self.channels[-1], X.shape[1], final_bias=False)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate)
        self.train()

        return self
    
    def _generate(self, z=None, n_topics=None):
        if n_topics is None:
            n_topics = self.n_topics_

        tw_probs = self._decoder(self._tw_rnn(n_layers=n_topics))

        if z is not None:
            dt_probs = F.softmax(z@self._dt_rnn(n_layers=n_topics).T, -1)
            X = dt_probs@tw_probs

            return X
        return tw_probs
    
    def _evaluate(self, X):
        z, kld = self._encoder(X, return_kld=True)
        X_k, X_K = self._generate(z, self.n_topics_ - 1), self._generate(z)
        loss_k = (X_k - X).square().sum(-1)/X.shape[0]
        loss_K = (X_K - X).square().sum(-1)/X.shape[0]

        if (loss_k - loss_K).sum()/loss_K.sum() > self.topic_rate:
            self.n_topics_ += 1

        loss = loss_K.sum() + self.kld_scale*kld

        return loss
    
    def _step(self):
        loss = 0.

        for x in self._loader:
            x_loss = self._evaluate(x)
            x_loss.backward()
            loss += x_loss.item()

        self._optim.step()
        self._optim.zero_grad()
        self.topic_log_.append(self.n_topics_)

        return loss
    
    def _predict(self, X, eval=True):
        if eval:
            self.eval()

        topics = (X@self._generate().T).argmax(-1).detach()

        return topics
