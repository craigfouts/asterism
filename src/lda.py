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
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from tqdm import tqdm
from base import HotTopic
from utils import kmeans, relabel

class GibbsLDA(HotTopic):
    def __init__(self, n_topics=3, *, vocab_size=10, doc_size=10, dt_prior=1., tw_prior=1., desc='LDA', random_state=None):
        super().__init__(desc, random_state)

        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.doc_size = doc_size
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
        self._dt_counts = np.eye(self.n_topics)[topics.reshape(*self._docs.shape)].sum(1)
        self._tw_counts = (topics == np.arange(self.n_topics)[None].T)@np.eye(self.vocab_size)[self._words]

        return self
    
    def _query(self, index):
        doc = index//self._docs.shape[1]
        word = self._words[index]
        topic = self._topics[self._step_n - 1, index]

        return doc, word, topic
    
    def _decrement(self, doc, word, topic, return_counts=False):
        self._dt_counts[doc, topic] -= 1
        self._tw_counts[topic, word] -= 1

        if return_counts:
            return self._dt_counts, self._tw_counts
    
    def _increment(self, doc, word, topic, return_counts=False):
        self._dt_counts[doc, topic] += 1
        self._tw_counts[topic, word] += 1

        if return_counts:
            return self._dt_counts, self._tw_counts
        
    def _sample(self, doc, word, return_probs=False):
        dt_ratio = self._dt_counts[doc] + self.dt_prior
        dt_ratio /= dt_ratio.sum()
        tw_ratio = self._tw_counts[:, word] + self.tw_prior
        tw_ratio /= (self._tw_counts + self.tw_prior).sum(-1)
        probs = dt_ratio*tw_ratio
        probs /= probs.sum()
        topic = self.random_state_.choice(self.n_topics, p=probs)

        return (topic, probs) if return_probs else topic
    
    def _step(self):
        indices = self.random_state_.permutation(self._words.shape[0])
        likelihood = 0

        for index in indices:
            doc, word, topic = self._query(index)
            self._decrement(doc, word, topic)
            new_topic, probs = self._sample(doc, word, return_probs=True)
            self._increment(doc, word, new_topic)
            self._topics[self._step_n, index] = new_topic
            likelihood += probs[new_topic]

        return likelihood
    
    def _predict(self, y=None):
        burn_in = self._n_steps//2
        word_topics = mode(self._topics[burn_in:]).mode
        doc_topics = mode(word_topics.reshape(*self._docs.shape), -1).mode
        topics = relabel(doc_topics, y)
        
        return topics

class PyroLDA(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_topics=5, vocab_size=10, document_size=10, document_prior=None, topic_prior=None):
        pyro.clear_param_store()
        super().__init__()

        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.document_size = document_size
        self.document_prior = torch.ones(n_topics)/n_topics if document_prior is None else document_prior
        self.topic_prior = torch.ones(vocab_size)/vocab_size if topic_prior is None else topic_prior

        self.corpus = None
        self.topic_posterior = None
        self.doc_posterior = None
        self.batch_size = None
        self.optimizer = None
        self.elbo = None
        self.svi = None
        self.labels_ = None
        self.log = []

    def build(self, data, learning_rate=1e-1):
        knn = torch.cdist(data, data).topk(self.document_size, largest=False).indices
        self.corpus = kmeans(data[:, 3:], self.vocab_size, verbosity=0)[knn].T
        self.optimizer = Adam({'lr': learning_rate})
        self.elbo = TraceEnum_ELBO(max_plate_nesting=2)
        self.svi = SVI(self.model, self.guide, self.optimizer, self.elbo)

        return self
    
    def model(self, data):
        with pyro.plate('topics', self.n_topics):
            topic_words = pyro.sample('topic_words', Dirichlet(self.topic_prior))

        with pyro.plate('documents', data.shape[1], self.batch_size) as mask:
            data = data[:, mask]
            document_topics = pyro.sample('document_topics', Dirichlet(self.document_prior))

            with pyro.plate('words', data.shape[0]):
                word_topics = pyro.sample('word_topics', Categorical(document_topics), infer={'enumerate': 'parallel'})
                pyro.sample('document_words', Categorical(topic_words[word_topics]), obs=data)

        return self

    def guide(self, data):
        self.topic_posterior = pyro.param('topic_posterior', lambda: torch.ones(self.n_topics, self.vocab_size), constraint=constraints.greater_than(.5))
        self.document_posterior = pyro.param('document_posterior', lambda: torch.ones(data.shape[1], self.n_topics), constraint=constraints.greater_than(.5))

        with pyro.plate('topics', self.n_topics):
            pyro.sample('topic_words', Dirichlet(self.topic_posterior))

        with pyro.plate('documents', data.shape[1], self.batch_size) as mask:
            data = data[:, mask]
            pyro.sample('document_topics', Dirichlet(self.document_posterior[mask]))

        return self

    def fit(self, data, n_steps=1000, learning_rate=1e-1, batch_size=100, verbosity=1, description='LDA'):
        self.build(data, learning_rate)
        self.batch_size = batch_size
        
        for _ in tqdm(range(n_steps), description) if verbosity == 1 else range(n_steps):
            loss = self.svi.step(self.corpus)
            self.log.append(loss)

        self.labels_ = relabel(pyro.sample('document_topics', Dirichlet(self.document_posterior)).argmax(-1))

        return self
    
    def transform(self, _=None):
        probabilities = pyro.sample('document_topics', Dirichlet(self.document_posterior))

        return probabilities
