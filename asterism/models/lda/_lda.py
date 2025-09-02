'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
import pyro
import pyro.distributions.constraints as constraints
import torch
from pyro.distributions import Categorical, Dirichlet
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from scipy.spatial.distance import cdist
from scipy.stats import mode
from ...base import Keynote
from ...utils import kmeans

class GibbsLDA(Keynote):
    def __init__(self, n_topics=3, *, doc_size=32, vocab_size=32, dt_prior=1., tw_prior=1., desc='LDA', seed=None):
        super().__init__(desc, seed)

        self.n_topics = n_topics
        self.doc_size = doc_size
        self.vocab_size = vocab_size
        self.dt_prior = dt_prior
        self.tw_prior = tw_prior

        self._n_steps = 100
        
    def _build(self, X):
        knn = cdist(X, X).argsort(-1)[:, :self.doc_size]
        self._docs = kmeans(X, self.vocab_size, verbosity=0)[knn]
        self._words = self._docs.flatten()
        topics = self.random_state_.choice(self.n_topics, self._words.shape[0])
        self._topics = np.zeros((self._n_steps, self._words.shape[0]), dtype=np.int32)
        self._topics[-1:] = topics
        self._dt_post = np.eye(self.n_topics)[topics.reshape(*self._docs.shape)].sum(1)
        self._tw_post = (topics == np.arange(self.n_topics)[None].T)@np.eye(self.vocab_size)[self._words]

        return self
    
    def _query(self, idx):
        doc = idx//self._docs.shape[1]
        topic = self._topics[self._step_n - 1, idx]
        word = self._words[idx]

        return doc, topic, word
    
    def _decrement(self, doc, topic, word, return_posts=False):
        self._dt_post[doc, topic] -= 1
        self._tw_post[topic, word] -= 1

        if return_posts:
            return self._dt_post, self._tw_post
        return self
    
    def _increment(self, doc, topic, word, return_posts=False):
        self._dt_post[doc, topic] += 1
        self._tw_post[topic, word] += 1

        if return_posts:
            return self._dt_post, self._tw_post
        return self
        
    def _sample(self, doc, word, return_probs=False):
        dt_probs = self._dt_post[doc] + self.dt_prior
        dt_probs /= dt_probs.sum()
        tw_probs = self._tw_post[:, word] + self.tw_prior
        tw_probs /= (self._tw_post + self.tw_prior).sum(-1)
        probs = dt_probs*tw_probs
        probs /= probs.sum()
        topic = self.random_state_.choice(self.n_topics, p=probs)

        if return_probs:
            return topic, probs
        return topic
    
    def _step(self):
        idx = self.random_state_.permutation(self._words.shape[0])
        likelihood = 0

        for i in idx:
            doc, topic, word = self._query(i)
            self._decrement(doc, topic, word)
            new_topic, probs = self._sample(doc, word, return_probs=True)
            self._increment(doc, new_topic, word)
            self._topics[self._step_n, i] = new_topic
            likelihood += probs[new_topic]

        return likelihood
    
    def _predict(self):
        burn_in = self._n_steps//2
        word_topics = mode(self._topics[burn_in:]).mode
        doc_topics = mode(word_topics.reshape(*self._docs.shape), -1).mode
        
        return doc_topics

class PyroLDA(Keynote):
    def __init__(self, n_topics=3, *, doc_size=32, vocab_size=32, dt_prior=1., tw_prior=1., optim='adam', desc='LDA'):
        pyro.clear_param_store()
        super().__init__(desc)

        self.n_topics = n_topics
        self.doc_size = doc_size
        self.vocab_size = vocab_size
        self.dt_prior = dt_prior
        self.tw_prior = tw_prior
        self.optim = optim

        self._n_steps = 1000

    def _build(self, X, learning_rate=1e-1, batch_size=128):
        knn = torch.cdist(X, X).topk(self.doc_size, largest=False).indices
        self._docs = kmeans(X, self.vocab_size, verbosity=0)[knn].T
        self._optim = Adam({'lr': learning_rate})
        self._elbo = TraceEnum_ELBO(max_plate_nesting=2)
        self._svi = SVI(self._model, self._guide, self._optim, self._elbo)
        self._batch_size = batch_size

        return self
    
    def _model(self, X):
        with pyro.plate('topics', self.n_topics):
            tw_dists = pyro.sample('tw_dists', Dirichlet(self.tw_prior*torch.ones(self.vocab_size)))

        with pyro.plate('docs', X.shape[1], self._batch_size) as mask:
            dt_dists = pyro.sample('dt_dists', Dirichlet(self.dt_prior*torch.ones(self.n_topics)))

            with pyro.plate('words', X.shape[0]):
                labels = pyro.sample('labels', Categorical(dt_dists), infer={'enumerate': 'parallel'})
                pyro.sample('values', Categorical(tw_dists[labels]), obs=X[:, mask])

        return self
    
    def _guide(self, X):
        dt_lambda = lambda: torch.ones((X.shape[1], self.n_topics))
        tw_lambda = lambda: torch.ones((self.n_topics, self.vocab_size))
        self._dt_post = pyro.param('dt_post', dt_lambda, constraint=constraints.greater_than(.5))
        self._tw_post = pyro.param('tw_post', tw_lambda, constraint=constraints.greater_than(.5))

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
