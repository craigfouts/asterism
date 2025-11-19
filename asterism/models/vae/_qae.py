'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from torch import nn
from ...core import Asterism
from ...utils.nets import OPTIMS, Encoder, MLP
from ...utils.sugar import attrmethod

__all__ = [
    'VQAE'
]
        
class VQAE(Asterism, nn.Module):
    @attrmethod
    def __init__(self, max_topics=100, *, channels=(128, 32), optim='adam', desc='VQAE', seed=None):
        super().__init__(desc, seed)

        self._n_steps = 100
        
    def _build(self, X, learning_rate=1e-2, weight_decay=1e-2):
        mask = torch.randperm(X.shape[0])[:self.max_topics]
        out_channels = (self.channels,) if isinstance(self.channels, int) else self.channels
        self._encoder = MLP(in_channels := X.shape[1], *out_channels, norm_layer='batch', act_layer='relu')
        self._decoder = MLP(*out_channels[::-1], in_channels, norm_layer='batch', act_layer='relu')
        self._codebook = nn.Parameter(self._encoder(X[mask]), requires_grad=True)
        self._optim = OPTIMS[self.optim](self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train()

        return self
    
    def _quantize(self, Z, Z_grad=False, C_grad=False, return_loss=False):
        embeddings = Z if Z_grad else Z.detach()
        codebook = self._codebook if C_grad else self._codebook.detach()
        cdist = (embeddings[:, None] - codebook[None]).square().sum(-1)
        topics = cdist.argmin(-1)
        
        if return_loss:
            loss = cdist[torch.arange(topics.shape[0]), topics].sum()

            return topics, loss
        return topics
    
    def _evaluate(self, X):
        Z = self._encoder(X)
        topics, Z_loss = self._quantize(Z, Z_grad=True, return_loss=True)
        _, C_loss = self._quantize(Z, C_grad=True, return_loss=True)
        X_ = self._decoder(self._codebook[topics])
        loss = Z_loss + C_loss + (X_ - X).square().sum().sqrt()

        return loss
    
    def _step(self, X):
        (loss := self._evaluate(X)).backward()  # TODO: add DataLoader
        self._optim.step()
        self._optim.zero_grad()

        return loss.item()
    
    def _predict(self, X, eval=True):
        if eval:
            self.eval()

        topics = self._quantize(self._encoder(X))

        return topics
    
# TODO
# class VQVAE(HotTopic, nn.Module):
#     def __init__(self, max_topics=100, *, channels=(64, 32), kld_scale=.1, optim='adam', desc='VQVAE', random_state=None):
#         super().__init__(desc, random_state)

#         self.max_topics = max_topics
#         self.channels = channels
#         self.kld_scale = kld_scale
#         self.optim = optim

#         self._n_steps = 100
        
#     def _build(self, X, learning_rate=1e-2, weight_decay=1e-2):  # TODO: add batch_size
#         n_samples, in_channels = X.shape
#         out_channels = (self.channels,) if isinstance(self.channels, int) else self.channels
#         self._encoder = Encoder(in_channels, *out_channels, norm_layer='batch', act_layer='relu')
#         self._decoder = MLP(*out_channels[::-1], in_channels, norm_layer='batch', act_layer='relu')
#         mask = torch.randperm(n_samples)[:self.max_topics]
#         self._codebook = nn.Parameter(self._encoder(X[mask]), requires_grad=True)
#         self._optim = OPTIM[self.optim](self.parameters(), lr=learning_rate, weight_decay=weight_decay)
#         self.train()

#         return self
    
#     def _quantize(self, z, z_grad=False, c_grad=False, return_loss=False):
#         embeddings = z if z_grad else z.detach()
#         codebook = self._codebook if c_grad else self._codebook.detach()
#         cdist = (embeddings[:, None] - codebook[None]).square().sum(-1)
#         topics = cdist.argmin(-1)
        
#         if return_loss:
#             loss = cdist[torch.arange(topics.shape[0]), topics].sum()

#             return topics, loss
#         return topics
    
#     def _evaluate(self, X, z):  # TODO: should be self-contained; shouldn't need to pass z (same for QAE)
#         topics, z_loss = self._quantize(z, z_grad=True, return_loss=True)
#         _, c_loss = self._quantize(z, c_grad=True, return_loss=True)
#         X_ = self._decoder(self._codebook[topics])
#         loss = z_loss + c_loss + (X_ - X).square().sum().sqrt()

#         return loss
    
#     def _step(self, X):
#         z, kld = self._encoder(X, return_kld=True)
#         loss = self._evaluate(X, z) + self.kld_scale*kld
#         loss.backward()
#         self._optim.step()
#         self._optim.zero_grad()

#         return loss.item()
    
#     def _predict(self, X):
#         self.eval()
#         z = self._encoder(X)
#         topics = self._quantize(z)

#         return topics
