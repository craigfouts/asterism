'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from tqdm import tqdm
from ...core import AsterismSpatial
from ...utils import normalize, relabel
from ...utils.sugar import buildmethod

__all__ = [
    'GibbsSLDA'
]

class GibbsSLDA(AsterismSpatial):
    def __init__(self, n_topics=3, *, n_docs=None, doc_size=4, data_size=4, vocab_size=32, dt_prior=1., tw_prior=1., desc='SLDA', seed=None):
        super().__init__(desc, seed)

        self.n_topics = n_topics
        self.n_docs = n_docs
        self.doc_size = doc_size
        self.data_size = data_size
        self.vocab_size = vocab_size
        self.dt_prior = dt_prior
        self.tw_prior = tw_prior

        self._n_steps = 100  # FIXME

    def _build_docs(self, locs, in_place=True):
        sections, docs = np.unique(locs[:, 0]), []    

        for s in sections:
            n_samples = (s_mask := locs[:, 0] == s).sum()
            s_docs = self.n_docs if self.n_docs is not None else n_samples//4
            s_idx = self._state.permutation(n_samples)[:s_docs]
            s_locs = (d_locs := locs[s_mask, :3][s_idx])[:, -2:]
            s_cdists = cdist(s_locs, s_locs, 'sqeuclidean')
            s_vars = np.sort(s_cdists, -1)[:, self.doc_size]
            docs.append(np.concat([d_locs, s_vars[:, None]], -1))
        
        docs = np.concat(docs, 0)

        if in_place:
            self._docs = docs

        return docs
    
    def _build_data(self, X, locs, in_place=True):
        sections, data = np.unique(locs[:, 0]), []

        for s in sections:
            s_locs = locs[s_mask := locs[:, 0] == s, -2:]
            s_cdists = cdist(s_locs, s_locs, 'sqeuclidean')
            s_vars = np.sort(s_cdists, -1)[:, self.data_size]
            gauss = np.exp(-s_cdists/(2*s_vars))/(np.sqrt(2*np.pi*s_vars))
            data.append(gauss@X[s_mask])

        data = np.concat(data, 0)

        if in_place:
            self._data = data

        return data

    def _build(self, X, locs):
        n_docs, _ = len(self._build_docs(self._locs)), self._build_data(X, self._locs)
        self.words_ = KMeans(self.vocab_size, random_state=self._state).fit_predict(self._data)
        self.docs_, self.topics_ = np.zeros((2, self._n_steps, n_samples := len(X)), dtype=np.int32)
        self.docs_[-1:] = self._state.choice(n_docs, n_samples)
        self.topics_[-1:] = self._state.choice(self.n_topics, n_samples)
        doc_range, topic_range = np.arange(n_docs)[:, None], np.arange(self.n_topics)[:, None]
        self.dt_post_ = (self.docs_[-1] == doc_range)@np.eye(self.n_topics)[self.topics_[-1]]
        self.tw_post_ = (self.topics_[-1] == topic_range)@np.eye(self.vocab_size)[self.words_]

        return self
    
    def _query(self, idx):
        loc = self._locs[idx]
        doc = self.docs_[self._step_n - 1, idx]
        topic = self.topics_[self._step_n - 1, idx]
        word = self.words_[idx]

        return loc, doc, topic, word
    
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
    
    def _sample_doc(self, loc, topic, return_probs=False):
        mask = self._docs[:, 0] == loc[0]
        cdists = cdist(loc[None, 1:], self._docs[mask, 1:3], 'sqeuclidean')[0]
        wd_probs = np.exp(-cdists/(2*self._docs[mask, -1]))
        dt_probs = self.dt_post_[mask, topic] + self.dt_prior
        dt_probs /= (self.dt_post_[mask] + self.dt_prior).sum(-1)
        probs = normalize(wd_probs*dt_probs)
        doc = self._state.choice(mask.sum(), p=probs)

        if return_probs:
            return doc, probs
        return doc

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
            loc, doc, topic, word = self._query(i)
            self._decrement(doc, topic, word)
            new_doc, doc_probs = self._sample_doc(loc, topic, return_probs=True)
            new_topic, topic_probs = self._sample_topic(doc, word, return_probs=True)
            self._increment(new_doc, new_topic, word)
            self.docs_[self._step_n, i] = new_doc
            self.topics_[self._step_n, i] = new_topic
            likelihood += doc_probs[new_doc] + topic_probs[new_topic]

        return likelihood

    def _predict(self):
        burn_in = self._n_steps//4
        topics = mode(self.topics_[-burn_in:]).mode

        return topics
