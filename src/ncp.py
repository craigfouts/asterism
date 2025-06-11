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
    wc_channels : int | tuple | list, default=256
        Architecture of the within-cluster invariant representation encoder.
    bc_channels : int | tuple | list, default=512
        Architecture of the between-cluster invariant representation encoder.
    lp_channels : int | tuple | list, default=1
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

    def __init__(self, in_channels, wc_channels=256, bc_channels=512, lp_channels=1, activation='prelu'):
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
        """Initializes class attributes and invariant representations.
        
        Parameters
        ----------
        data : Tensor
            Feature values for each sample in each batch.

        Returns
        -------
        self
            I return therefore I am.
        """
        
        self.batch_size = data.shape[0] if len(data.shape) > 2 else 1
        (self.n_samples, n_features), self.n_topics = data.shape[-2:], 1
        data = data.reshape(self.batch_size*self.n_samples, n_features)
        self.wc = self.wc_model(data).view(self.batch_size, self.n_samples, self.wc_channels[-1])
        self.us = self.us_model(data).view(self.batch_size, self.n_samples, self.wc_channels[-1])
        self.WC = torch.zeros((self.batch_size, 1, self.wc_channels[-1]))
        self.WC[:, 0], self.US = self.wc[:, 0], self.us[:, 2:].sum(1)

        return self
    
    def update(self, sample, topics):
        """
        """
        
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
    
    def logprob(self, sample, topic):
        if topic == self.n_topics:
            WC = torch.cat((self.WC, self.wc[:, sample].unsqueeze(1)), 1)
        else:
            WC = self.WC.clone()
            WC[:, topic] += self.wc[:, sample]

        WC = WC.view((self.batch_size*WC.shape[1], self.wc_channels[-1]))
        BC = self.bc_model(WC)
        BC = BC.view(self.batch_size, BC.shape[0]//self.batch_size, self.bc_channels[-1]).sum(1)
        logprob = self.lp_model(torch.cat((BC, self.US), 1)).squeeze()

        return logprob
    
    def sample(self, sample, normalize=True, return_topics=True):
        logprobs = torch.zeros(self.batch_size, self.n_topics + 1)

        for k in range(self.n_topics + 1):
            logprobs[:, k] = self.logprob(sample, k)

        if normalize:
            m, _ = logprobs.max(1, keepdim=True)
            logprobs = logprobs - m - (logprobs - m).exp().sum(1, keepdim=True).log()

        if return_topics:
            topics = torch.multinomial(logprobs.exp(), 1).T[0]

            return topics, logprobs
        
        return logprobs
    
    def evaluate(self, data, labels):
        self.build(data)
        loss = 0

        for i in range(2, self.n_samples):
            self.n_topics = self.update(i, labels)
            loss -= self.sample(i, return_topics=False)[:, labels[i]].mean()

        return loss
    
    def forward(self, data):
        self.build(data)
        topics = torch.zeros(data.shape[1], dtype=torch.int32)

        for i in range(2, self.n_samples):
            self.n_topics = self.update(i, topics)
            topics[i] = torch.mode(self.sample(i)[0])[0].item()

        return topics
    
class NCP(nn.Module):
    """Implementation of a neural clustering process model. Based on methods
    proposed by Ari Pakman, Yueqi Wang, Catalin Mitelut, JinHyung Lee, and Liam
    Paninski.
    
    https://proceedings.mlr.press/v119/pakman20a.html

    Parameters
    ----------
    wc_channels : int | tuple | list, default=256
        Architecture of the within-cluster invariant encoder.
    bc_channels : int | tuple | list, default=512
        Architecture of the between-cluster invariant encoder.
    lp_channels : int | tuple | list, default=1
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

    def __init__(self, wc_channels=256, bc_channels=512, lp_channels=1, activation='prelu'):
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
    
    def fit(self, data_maker, n_steps=200, n_permutations=6, learning_rate=1e-4, weight_decay=1e-2, verbosity=1, description='NCP'):
        self.build(data_maker.n_features, learning_rate, weight_decay)

        for _ in tqdm(range(n_steps), description) if verbosity == 1 else range(n_steps):
            (data, labels), nll = data_maker.make(), 0

            for _ in range(n_permutations):
                nll += self.evaluate(*shuffle(data, labels, sort=True), grad=True)

            self.step(nll/data.shape[1])

        return self
    
    def forward(self, data):
        topics = self.encoder(data)

        return topics
