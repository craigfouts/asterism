"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import torch
import torch.nn as nn
from nets import MLP
from torch.optim import Adam
from tqdm import tqdm
from utils import shuffle

class Encoder(nn.Module):
    """Implementation of a neural clustering process encoder. Based on methods
    proposed by Ari Pakman, Yueqi Wang, Catalin Mitelut, JinHyung Lee, and Liam
    Paninski.
    
    https://proceedings.mlr.press/v119/pakman20a.html

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    wc_channels : int | tuple | list, default=(64, 64)
        Architecture of the within-cluster invariant representation encoder.
    bc_channels : int | tuple | list, default=(128, 128)
        Architecture of the between-cluster invariant representation encoder.
    lp_channels : int | tuple | list, default=(32, 32)
        Architecture of the topic log probability encoder.
    activation : str, default='prelu'
        Hidden activation function.
    
    Attributes
    ----------
    wc_model : MLP
        Within-cluster invariant representation encoder.
    us_model : MLP
        Unassigned sample invariant representation encoder.
    bc_model : MLP
        Between-cluster invariant representation encoder.
    lp_model : MLP
        Topic log probability encoder.

    Usage
    -----
    >>> model = Encoder(in_channels, *args, **kwargs)
    >>> topics = model(data)
    """

    def __init__(self, in_channels, wc_channels=(64, 64), bc_channels=(128, 128), lp_channels=(32, 32), activation='prelu'):
        super().__init__()

        self.in_channels = in_channels
        self.wc_channels = (wc_channels,) if isinstance(wc_channels, int) else wc_channels
        self.bc_channels = (bc_channels,) if isinstance(bc_channels, int) else bc_channels
        self.lp_channels = (lp_channels,) if isinstance(lp_channels, int) else lp_channels
        self.lp_channels += (1,)*(self.lp_channels[-1] != 1)

        self.wc_model = MLP(in_channels, *self.wc_channels, activation=activation)
        self.us_model = MLP(in_channels, *self.wc_channels, activation=activation)
        self.bc_model = MLP(self.wc_channels[-1], *self.bc_channels, activation=activation)
        self.lp_model = MLP(self.wc_channels[-1] + self.bc_channels[-1], *self.lp_channels, activation=activation, out_bias=False)

    def build(self, data):
        self.batch_size = data.shape[0] if len(data.shape) > 2 else 1
        self.n_samples, self.n_topics = data.shape[-2], 1
        self.wc, self.us = self.wc_model(data), self.us_model(data)
        self.WC = torch.zeros((self.batch_size, 1, self.wc_channels[-1]))
        self.WC[:, 0], self.US = self.wc[:, 0], self.us[:, 2:].sum(1)

        return self
    
    def update(self, sample, topics):
        n_topics = topics[:sample].unique().shape[0]

        if n_topics == self.n_topics:
            self.WC[:, topics[sample - 1]] += self.wc[:, sample - 1]
        else:
            self.WC = torch.cat((self.WC, self.wc[:, sample - 1].unsqueeze(1)), 1)

        if sample == self.n_samples - 1:
            self.US = torch.zeros(self.batch_size, self.wc_channels[-1])
        else:
            self.US -= self.us[:, sample]

        return n_topics
    
    def logprobs(self, sample):
        logprobs = torch.zeros(self.batch_size, self.n_topics + 1)
        topic_range = torch.arange(self.n_topics)
        US_k = self.US.repeat(self.n_topics, 1, 1)
        WC_k = self.WC.repeat(self.n_topics, 1, 1, 1)
        WC_k[topic_range, :, topic_range] += self.wc[:, sample]
        WC_K = torch.cat((self.WC, self.wc[:, sample].unsqueeze(1)), 1)
        BC_k, BC_K = self.bc_model(WC_k).sum(2), self.bc_model(WC_K).sum(1)
        logprobs[:, :-1] = self.lp_model(torch.cat((US_k, BC_k), -1))[..., 0].T
        logprobs[:, -1] = self.lp_model(torch.cat((self.US, BC_K), 1)).squeeze()

        return logprobs
    
    def sample(self, sample, normalize=True, return_topics=True):
        logprobs = self.logprobs(sample)

        if normalize:
            m, _ = logprobs.max(1, keepdim=True)
            logprobs = logprobs - m - (logprobs - m).exp().sum(1, keepdim=True).log()

        if return_topics:
            topics = torch.multinomial(logprobs.exp(), 1).T[0]

            return topics
        
        return logprobs
    
    def evaluate(self, data, labels):
        self.build(data)
        nll = 0

        for i in range(2, self.n_samples):
            self.n_topics = self.update(i, labels)
            nll -= self.sample(i, return_topics=False)[:, labels[i]].mean()

        return nll
    
    def forward(self, data):
        self.build(data)
        topics = torch.zeros(data.shape[1], dtype=torch.int32)

        for i in range(2, self.n_samples):
            self.n_topics = self.update(i, topics)
            topics[i] = torch.mode(self.sample(i))[0].item()

        return topics
    
class NCP(nn.Module):
    """Implementation of a neural clustering process model. Based on methods
    proposed by Ari Pakman, Yueqi Wang, Catalin Mitelut, JinHyung Lee, and Liam
    Paninski.
    
    https://proceedings.mlr.press/v119/pakman20a.html

    Parameters
    ----------
    wc_channels : int | tuple | list, default=(128, 128)
        Architecture of the within-cluster invariant encoder.
    bc_channels : int | tuple | list, default=(512, 512)
        Architecture of the between-cluster invariant encoder.
    lp_channels : int | tuple | list, default=(128, 128)
        Architecture of the topic log probability encoder.
    activation : str, default='prelu'
        Hidden activation function.

    Attributes
    ----------
    encoder : Encoder
        Neural clustering process encoder.
    optimizer : Optimzer
        Optimization handler.
    nll_log : list
        Record of the total negative log likelihood for each step.
    
    Ussage
    ------
    >>> model = NCP(*args, **kwargs).fit(generator, *args, **kwargs)
    >>> topics = model(data)
    """

    def __init__(self, wc_channels=(128, 128), bc_channels=(512, 512), lp_channels=(128, 128), activation='prelu'):
        super().__init__()

        self.wc_channels = wc_channels
        self.bc_channels = bc_channels
        self.lp_channels = lp_channels
        self.activation = activation

        self.encoder = None
        self.optimizer = None
        self.nll_log = []
    
    def build(self, in_channels, learning_rate=1e-4, weight_decay=1e-2):
        self.encoder = Encoder(in_channels, self.wc_channels, self.bc_channels, self.lp_channels, self.activation)
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        return self
    
    def step(self, nll=None):
        self.optimizer.step()
        self.optimizer.zero_grad()

        if nll is not None:
            self.nll_log.append(nll)

        return self
    
    def evaluate(self, data, labels, grad=False):
        nll = self.encoder.evaluate(data, labels)

        if grad:
            nll.backward()

        return nll.item()
    
    def fit(self, data, labels, n_steps=200, n_permutations=6, learning_rate=1e-4, weight_decay=1e-2, batch_size=16, n_samples=64, verbosity=1, description='NCP'):
        self.build(data.shape[-1], learning_rate, weight_decay)
        self.batch_size = max(data.shape[0], batch_size)

        for _ in tqdm(range(n_steps), description) if verbosity == 1 else range(n_steps):
            mask, nll = torch.randperm(data.shape[0])[:batch_size], 0

            for _ in range(n_permutations):
                X, y = shuffle(data[mask], labels, sort=True, cut=n_samples)
                nll += self.evaluate(X, y, grad=True)

            self.step(nll)
    
    def forward(self, data):
        if data.shape[0] == 1 or len(data.shape) == 2:
            data = data.repeat(self.batch_size, 1, 1)

        topics = self.encoder(data)
        
        return topics
