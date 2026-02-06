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
from sklearn.cluster import KMeans
from ...core import Asterism
from ...utils import kmeans, normalize
from ...utils.sugar import attrmethod

__all__ = [
    'GibbsLDA',
    'PyroLDA'
]

class GibbsLDA(Asterism):
    @attrmethod
    def __init__(self, n_topics=3, *, doc_size=32, vocab_size=32, dt_prior=1., tw_prior=1., desc='LDA', seed=None):
        super().__init__(desc, seed)

        self._n_steps = 100
        
    def _build(self, X):
        edges = cdist(X, X).argsort(-1)[:, :self.doc_size]
        self.docs_ = KMeans(self.vocab_size, random_state=self._state).fit_predict(X)[edges]
        self.words_, topic_range = self.docs_.flatten(), np.arange(self.n_topics)[:, None]
        self.topics_ = np.zeros((self._n_steps, n_words := len(self.words_)), dtype=np.int32)
        self.topics_[-1] = self._state.choice(self.n_topics, n_words)
        self.dt_post_ = np.eye(self.n_topics)[self.topics_[-1].reshape(*self.docs_.shape)].sum(1)
        self.tw_post_ = (self.topics_[-1] == topic_range)@np.eye(self.vocab_size)[self.words_]

        return self
    
    def _query(self, idx):
        doc = idx//self.docs_.shape[1]
        topic = self.topics_[self._step_n - 1, idx]
        word = self.words_[idx]

        return doc, topic, word
    
    def _decrement(self, doc, topic, word, return_posts=False):
        self.dt_post_[doc, topic] -= 1
        self.tw_post_[topic, word] -= 1

        if return_posts:
            return self.dt_post_, self.tw_post_
        return self
    
    def _increment(self, doc, topic, word, return_posts=False):
        self.dt_post_[doc, topic] += 1
        self.tw_post_[topic, word] += 1

        if return_posts:
            return self.dt_post_, self.tw_post_
        return self
        
    def _sample_topic(self, doc, word, return_probs=False):
        dt_probs = normalize(self.dt_post_[doc] + self.dt_prior)
        tw_probs = self.tw_post_[:, word] + self.tw_prior
        tw_probs /= (self.tw_post_ + self.tw_prior).sum(-1)
        probs = normalize(dt_probs*tw_probs)
        topic = self._state.choice(self.n_topics, p=probs)

        if return_probs:
            return topic, probs
        return topic
    
    def _step(self):
        idx, likelihood = self._state.permutation(self.words_.shape[0]), 0

        for i in idx:
            doc, topic, word = self._query(i)
            self._decrement(doc, topic, word)
            new_topic, topic_probs = self._sample_topic(doc, word, return_probs=True)
            self._increment(doc, new_topic, word)
            self.topics_[self._step_n, i] = new_topic
            likelihood += topic_probs[new_topic]

        return likelihood
    
    def _predict(self):
        burn_in = self._n_steps//2
        topics = mode(self.topics_[burn_in:]).mode
        topics = mode(topics.reshape(*self.docs_.shape), -1).mode
        
        return topics

class PyroLDA(Asterism):
    def __init__(self, n_topics=3, *, doc_size=32, vocab_size=32, dt_prior=1., tw_prior=1., optim='adam', desc='LDA', seed=None):
        pyro.clear_param_store()
        super().__init__(desc, seed, torch_state=True)

        self.n_topics = n_topics
        self.doc_size = doc_size
        self.vocab_size = vocab_size
        self.dt_prior = dt_prior
        self.tw_prior = tw_prior
        self.optim = optim

        self._n_steps = 1000

    def _build(self, X, learning_rate=1e-1, batch_size=128):
        edges = torch.cdist(X, X).topk(self.doc_size, largest=False).indices
        self.docs_ = kmeans(X, self.vocab_size, verbosity=0, seed=self._state)[edges].T
        self._optim, self._batch_size = Adam({'lr': learning_rate}), batch_size
        self._elbo = TraceEnum_ELBO(max_plate_nesting=2)
        self._svi = SVI(self._model, self._guide, self._optim, self._elbo)

        return self
    
    def _model(self, X):
        with pyro.plate('topics', self.n_topics):
            tw_probs = pyro.sample('tw_probs', Dirichlet(self.tw_prior*torch.ones(self.vocab_size)))

        with pyro.plate('docs', X.shape[1], self._batch_size) as mask:
            dt_probs = pyro.sample('dt_probs', Dirichlet(self.dt_prior*torch.ones(self.n_topics)))

            with pyro.plate('words', X.shape[0]):
                labels = pyro.sample('labels', Categorical(dt_probs), infer={'enumerate': 'parallel'})
                pyro.sample('values', Categorical(tw_probs[labels]), obs=X[:, mask])

        return self
    
    def _guide(self, X):
        dt_prior = lambda: torch.ones((X.shape[1], self.n_topics))
        tw_prior = lambda: torch.ones((self.n_topics, self.vocab_size))
        self._dt_post = pyro.param('dt_post', dt_prior, constraint=constraints.greater_than(.5))
        self._tw_post = pyro.param('tw_post', tw_prior, constraint=constraints.greater_than(.5))

        with pyro.plate('topics', self.n_topics):
            pyro.sample('tw_probs', Dirichlet(self._tw_post))

        with pyro.plate('docs', X.shape[1], self._batch_size) as mask:
            pyro.sample('dt_probs', Dirichlet(self._dt_post[mask]))

        return self
    
    def _step(self):
        loss = self._svi.step(self.docs_)

        return loss
    
    def _predict(self):
        topics = pyro.sample('dt_probs', Dirichlet(self._dt_post)).argmax(-1).detach()

        return topics
