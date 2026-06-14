'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
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
from ...base import Asterism
from ...utils import kmeans, normalize
from ...utils.sugar import attrmethod, buildmethod

__all__ = [
    'GibbsLDA',
    'PyroLDA'
]

class GibbsLDA(Asterism):
    @attrmethod
    def __init__(self, n_topics, *, doc_size=32, vocab_size=16, dt_prior=1., tw_prior=1., desc='LDA', seed=None):
        super().__init__(desc, seed)

        self._n_steps = 50

    def _build(self, X, burn_in=-2):
        self._burn_in = self._n_steps//-burn_in if burn_in < 0 else burn_in
        edges = cdist(X, X, 'seuclidean').argsort(-1)[:, :self.doc_size]
        self.docs_ = kmeans(X, self.vocab_size, seed=self._state)[edges]
        self.words_, topic_range = self.docs_.flatten(), np.arange(self.n_topics)[:, None]
        self.topics_ = np.zeros((self._n_steps, n_words := self.words_.shape[0]), dtype=np.int32)
        self.topics_[-1] = self._state.choice(self.n_topics, n_words)
        self.dt_post_ = np.eye(self.n_topics)[self.topics_[-1].reshape(*self.docs_.shape)].sum(1)
        self.tw_post_ = (self.topics_[-1] == topic_range)@np.eye(self.vocab_size)[self.words_]

    def _query(self, idx):
        doc = idx//self.docs_.shape[1]
        topic = self.topics_[self._step_n - 1, idx]
        word = self.words_[idx]

        return doc, topic, word

    def _decrement(self, doc, topic, word):
        self.dt_post_[doc, topic] -= 1
        self.tw_post_[topic, word] -= 1

    def _increment(self, doc, topic, word):
        self.dt_post_[doc, topic] += 1
        self.tw_post_[topic, word] += 1

    def _sample_topic(self, doc, word):
        dt_probs = normalize(self.dt_post_[doc] + self.dt_prior)
        tw_probs = self.tw_post_[:, word] + self.tw_prior
        tw_probs /= (self.tw_post_ + self.tw_prior).sum(-1)
        probs = normalize(dt_probs*tw_probs)
        topic = self._state.choice(self.n_topics, p=probs)

        return topic, probs[topic]

    def _step(self):
        perm, prob = self._state.permutation(self.words_.shape[0]), 0

        for idx in perm:
            doc, topic, word = self._query(idx)
            self._decrement(doc, topic, word)
            topic_, topic_prob = self._sample_topic(doc, word)
            self._increment(doc, topic_, word)
            self.topics_[self._step_n, idx] = topic_
            prob += topic_prob

        return prob

    def _predict(self):
        topics = mode(self.topics_[self._burn_in:]).mode
        topics = mode(topics.reshape(*self.docs_.shape), -1).mode
        
        return topics

    def _display(self):
        super()._display('likelihood')

class PyroLDA(Asterism):
    @attrmethod
    def __init__(self, n_topics, *, doc_size=32, vocab_size=16, dt_prior=1., tw_prior=1., desc='LDA', seed=None):
        super().__init__(desc, seed)

        self._n_steps = 200

    def _build(self, X, learn_rate=1e-1, batch_size=-1):
        self._batch_size = X.shape[0]//-batch_size if batch_size < 0 else batch_size
        optim, elbo = Adam({'lr': learn_rate}), TraceEnum_ELBO(max_plate_nesting=2)
        self._svi = SVI(self._model, self._guide, optim, elbo)
        self._dt_prior = self.dt_prior*torch.ones([X.shape[0], self.n_topics])
        self._tw_prior = self.tw_prior*torch.ones([self.n_topics, self.vocab_size])
        edges = torch.cdist(X, X).topk(self.doc_size, largest=False).indices
        self.docs_ = kmeans(X, self.vocab_size, seed=self._state)[edges].T

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
        with pyro.plate('topics', self.n_topics):
            tw_post = pyro.param('tw_post', self._tw_prior, constraint=constraints.greater_than(.5))
            pyro.sample('tw_probs', Dirichlet(tw_post))

        with pyro.plate('docs', X.shape[1], self._batch_size) as mask:
            dt_post = pyro.param('dt_post', self._dt_prior, constraint=constraints.greater_than(.5))
            pyro.sample('dt_probs', Dirichlet(dt_post[mask]))

        return self

    def _step(self):
        loss = self._svi.step(self.docs_)

        return loss

    def _predict(self):
        dt_post = pyro.param('dt_post', self._dt_prior, constraint=constraints.greater_than(.5))
        topics = pyro.sample('dt_probs', Dirichlet(dt_post)).argmax(-1).detach()

        return topics
