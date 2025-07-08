"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import numpy as np
import pyro
import pyro.distributions.constraints as constraints
import torch
from pyro.distributions import Categorical, Dirichlet
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from scipy.spatial.distance import cdist
from scipy.stats import mode
from base import HotTopic
from utils import kmeans

class GibbsLDA(HotTopic):
    def __init__(self, n_topics=3, *, doc_size=10, vocab_size=10, tw_prior=.1, dt_prior=.1, desc='LDA', random_state=None):
        super().__init__(desc, random_state)

        self.n_topics = n_topics
        self.doc_size = doc_size
        self.vocab_size = vocab_size
        self.tw_prior = tw_prior
        self.dt_prior = dt_prior

        self._n_steps = 100
        
    def _build(self, X):
        knn = cdist(X, X).argsort(-1)[:, :self.doc_size]
        self._docs = kmeans(X, self.vocab_size, verbosity=0)[knn]
        self._words = self._docs.flatten()
        topics = self.random_state_.choice(self.n_topics, self._words.shape[0])
        self._topics = np.zeros((self._n_steps, self._words.shape[0]), dtype=np.int32)
        self._topics[-1:] = topics
        self._tw_counts = (topics == np.arange(self.n_topics)[None].T)@np.eye(self.vocab_size)[self._words]
        self._dt_counts = np.eye(self.n_topics)[topics.reshape(*self._docs.shape)].sum(1)

        return self
    
    def _query(self, index):
        topic = self._topics[self._step_n - 1, index]
        doc = index//self._docs.shape[1]
        word = self._words[index]

        return topic, doc, word
    
    def _decrement(self, topic, doc, word, return_counts=False):
        self._tw_counts[topic, word] -= 1
        self._dt_counts[doc, topic] -= 1

        if return_counts:
            return self._tw_counts, self._dt_counts
        return self
    
    def _increment(self, topic, doc, word, return_counts=False):
        self._tw_counts[topic, word] += 1
        self._dt_counts[doc, topic] += 1

        if return_counts:
            return self._tw_counts, self._dt_counts
        return self
        
    def _sample(self, doc, word, return_probs=False):
        tw_ratio = self._tw_counts[:, word] + self.tw_prior
        tw_ratio /= (self._tw_counts + self.tw_prior).sum(-1)
        dt_ratio = self._dt_counts[doc] + self.dt_prior
        dt_ratio /= dt_ratio.sum()
        probs = tw_ratio*dt_ratio
        probs /= probs.sum()
        topic = self.random_state_.choice(self.n_topics, p=probs)

        if return_probs:
            return topic, probs
        return topic
    
    def _step(self):
        indices = self.random_state_.permutation(self._words.shape[0])
        likelihood = 0

        for index in indices:
            topic, doc, word = self._query(index)
            self._decrement(topic, doc, word)
            new_topic, probs = self._sample(doc, word, return_probs=True)
            self._increment(new_topic, doc, word)
            self._topics[self._step_n, index] = new_topic
            likelihood += probs[new_topic]

        return likelihood
    
    def _predict(self):
        burn_in = self._n_steps//2
        labels = mode(self._topics[burn_in:]).mode
        topics = mode(labels.reshape(*self._docs.shape), -1).mode
        
        return topics

class PyroLDA(HotTopic):
    def __init__(self, n_topics=3, *, doc_size=10, vocab_size=10, tw_prior=.1, dt_prior=.1, optim='adam', desc='LDA'):
        pyro.clear_param_store()
        super().__init__(desc)

        self.n_topics = n_topics
        self.doc_size = doc_size
        self.vocab_size = vocab_size
        self.tw_prior = tw_prior
        self.dt_prior = dt_prior
        self.optim = optim

        self._n_steps = 1000
        self._tw_prior = self.tw_prior*torch.ones(self.vocab_size)
        self._dt_prior = self.dt_prior*torch.ones(self.n_topics)

    def _build(self, X, learning_rate=1e-1, batch_size=100):
        knn = torch.cdist(X, X).topk(self.doc_size, largest=False).indices
        self._docs = kmeans(X, self.vocab_size, verbosity=0)[knn].T
        self._optim = Adam({'lr': learning_rate})
        self._elbo = TraceEnum_ELBO(max_plate_nesting=2)
        self._svi = SVI(self._model, self._guide, self._optim, self._elbo)
        self._batch_size = batch_size

        return self
    
    def _model(self, X):
        with pyro.plate('topics', self.n_topics):
            tw_dists = pyro.sample('tw_dists', Dirichlet(self._tw_prior))

        with pyro.plate('docs', X.shape[1], self._batch_size) as mask:
            dt_dists = pyro.sample('dt_dists', Dirichlet(self._dt_prior))

            with pyro.plate('words', X.shape[0]):
                labels = pyro.sample('labels', Categorical(dt_dists), infer={'enumerate': 'parallel'})
                pyro.sample('values', Categorical(tw_dists[labels]), obs=X[:, mask])

        return self
    
    def _guide(self, X):
        self._tw_post = pyro.param('tw_post', lambda: torch.ones((self.n_topics, self.vocab_size)), constraint=constraints.greater_than(.5))
        self._dt_post = pyro.param('dt_post', lambda: torch.ones(X.shape[1], self.n_topics), constraint=constraints.greater_than(.5))

        with pyro.plate('topics', self.n_topics):
            pyro.sample('tw_dists', Dirichlet(self._tw_post))

        with pyro.plate('docs', X.shape[1], self._batch_size) as mask:
            pyro.sample('dt_dists', Dirichlet(self._dt_post[mask]))

        return self
    
    def _step(self):
        loss = self._svi.step(self._docs)

        return loss
    
    def _predict(self):
        labels = pyro.sample('dt_dists', Dirichlet(self._dt_post)).argmax(-1).detach()

        return labels
