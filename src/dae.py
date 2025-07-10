"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import torch
from torch import nn
from base import HotTopic
from nets import OPTIM, MLP
        
class VQAE(HotTopic, nn.Module):
    def __init__(self, max_topics=100, *, channels=(64, 32), optim='adam', desc='QAE', random_state=None):
        super().__init__(desc, random_state)

        self.max_topics = max_topics
        self.channels = channels
        self.optim = optim

        self._n_steps = 100
        
    def _build(self, X, learning_rate=1e-2, weight_decay=1e-2):
        in_channels = X.shape[-1]
        out_channels = (self.channels,) if isinstance(self.channels, int) else self.channels
        self._encoder = MLP(in_channels, *out_channels, norm_layer='batch', act_layer='relu')
        self._decoder = MLP(*out_channels[::-1], in_channels, norm_layer='batch', act_layer='relu')
        mask = torch.randperm(X.shape[0])[:self.max_topics]
        self._codebook = nn.Parameter(self._encoder(X)[mask], requires_grad=True)
        self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        return self
    
    def _quantize(self, z, z_grad=False, c_grad=False, return_loss=False):
        embeddings = z if z_grad else z.detach()
        codebook = self._codebook if c_grad else self._codebook.detach()
        cdist = (embeddings[:, None] - codebook[None]).square().sum(-1)
        topics = cdist.argmin(-1)
        
        if return_loss:
            loss = cdist[torch.arange(topics.shape[0]), topics].sum()

            return topics, loss
        return topics
    
    def _evaluate(self, X, z):
        topics, z_loss = self._quantize(z, z_grad=True, return_loss=True)
        _, c_loss = self._quantize(z, c_grad=True, return_loss=True)
        X_ = self._decoder(self._codebook[topics])
        loss = z_loss + c_loss + (X_ - X).square().sum().sqrt()

        return loss
    
    def _step(self, X):
        z = self._encoder(X)
        loss = self._evaluate(X, z)
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()

        return loss.item()
    
    def _predict(self, X):
        z = self._encoder(X)
        topics = self._quantize(z)

        return topics
