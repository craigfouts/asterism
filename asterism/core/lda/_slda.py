'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import mode
from ...base import Asterism
from ...utils import kmeans, normalize
from ...utils.sugar import attrmethod, buildmethod

__all__ = [
    'GibbsSLDA'
]

class GibbsSLDA(Asterism):
    @attrmethod
    def __init__(self, n_topics, *, n_docs=-8, doc_size=4., word_size=4., vocab_size=16, dt_prior=1., tw_prior=1., desc='SLDA', seed=None):
        super().__init__(desc, seed)

        self._n_steps = 200

    def _build_docs(self, locs):
        imgs, docs = np.unique(locs[:, 0]), []

        for img in imgs:
            n_pts = int((img_mask := locs[:, 0] == img).sum())
            n_docs = n_pts//-self.n_docs if self.n_docs < 0 else self.n_docs
            img_perm = self._state.permutation(n_pts)[:n_docs]
            doc_locs = (img_locs := locs[img_mask][img_perm])[:, 1:]
            img_prox = cdist(doc_locs, doc_locs, 'sqeuclidean')
            img_vars = self.doc_size*np.sort(img_prox, -1)[:, 1].mean()
            docs.append(np.concat([img_locs, img_vars*np.ones([n_docs, 1])], -1))

        self._docs = np.concat(docs, 0)

    def _build_words(self, X, locs):
        imgs, words = np.unique(locs[:, 0]), []

        for img in imgs:
            img_locs = locs[img_mask := locs[:, 0] == img, 1:]
            img_prox = cdist(img_locs, img_locs, 'sqeuclidean')
            img_vars = self.doc_size*np.sort(img_prox, -1)[:, 1].mean()
            img_conv = np.exp(-img_prox/(2*img_vars))/(np.sqrt(2*np.pi*img_vars))
            words.append(img_conv@X[img_mask])

        self._words = np.concat(words, 0)

    @buildmethod
    def _build(self, X, locs, burn_in=-2):
        self._burn_in = self._n_steps//-burn_in if burn_in < 0 else burn_in
        self.words_ = kmeans(self._words, self.vocab_size, seed=self._state)
        self.docs_, self.topics_ = np.zeros([2, self._n_steps, n_pts := X.shape[0]], dtype=np.int32)
        self.docs_[-1:] = self._state.choice(n_docs := self._docs.shape[0], n_pts)
        self.topics_[-1:] = self._state.choice(self.n_topics, n_pts)
        doc_range, topic_range = np.arange(n_docs)[:, None], np.arange(self.n_topics)[:, None]
        self.dt_post_ = (self.docs_[-1] == doc_range)@np.eye(self.n_topics)[self.topics_[-1]]
        self.tw_post_ = (self.topics_[-1] == topic_range)@np.eye(self.vocab_size)[self.words_]

    def _query(self, idx):
        loc = self._locs[idx]
        doc = self.docs_[self._step_n - 1, idx]
        topic = self.topics_[self._step_n - 1, idx]
        word = self.words_[idx]

        return loc, doc, topic, word

    def _decrement(self, doc, topic, word):
        self.dt_post_[doc, topic] -= 1
        self.tw_post_[topic, word] -= 1

    def _increment(self, doc, topic, word):
        self.dt_post_[doc, topic] += 1
        self.tw_post_[topic, word] += 1

    def _sample_doc(self, loc, topic):
        n_docs = (mask := self._docs[:, 0] == loc[0]).sum()
        prox = cdist(loc[None, 1:], self._docs[mask, 1:3], 'sqeuclidean')[0]
        wd_probs = np.exp(-prox/(2*self._docs[mask, -1]))
        dt_probs = self.dt_post_[mask, topic] + self.dt_prior
        dt_probs /= (self.dt_post_[mask] + self.dt_prior).sum(-1)
        probs = normalize(wd_probs*dt_probs)
        doc = self._state.choice(n_docs, p=probs)

        return doc, probs[doc]

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
            loc, doc, topic, word = self._query(idx)
            self._decrement(doc, topic, word)
            doc_, doc_prob = self._sample_doc(loc, topic)
            topic_, topic_prob = self._sample_topic(doc, word)
            self._increment(doc_, topic_, word)
            self.docs_[self._step_n, idx] = doc_
            self.topics_[self._step_n, idx] = topic_
            prob += doc_prob + topic_prob

        return prob

    def _predict(self):
        topics = mode(self.topics_[self._burn_in:]).mode

        return topics

    def _display(self):
        super()._display('likelihood')

    def fit(self, X, locs, y=None, *args, **kwargs):
        super().fit(X, y, locs, *args, **kwargs)

        return self

    def fit_predict(self, X, locs, y=None, *args, **kwargs):
        super().fit(X, y, locs, *args, **kwargs)

        return self.labels_
