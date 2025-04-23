"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from tqdm import tqdm
from util import set_seed

def shuffle(documents, words, n_topics=5, n_words=50, return_counts=False):
    """Randomly assigns a topic to each sample.
    
    Parameters
    ----------
    documents : ndarray
        Document assignment for each sample.
    words : ndarray
        Word assignment for each sample.
    n_topics : int, default=5
        Number of discoverable topics.
    n_words : int, default=50
        Number of phenotypic words.
    return_counts : bool, default=False
        Whether to return assignment counts.

    Returns
    -------
    ndarray
        Topic assignments.
    ndarray
        Topic assignment counts.
    ndarray
        Document assignment counts.
    """
    
    n_samples, n_documents = words.shape[0], np.unique(documents).shape[0]
    topics = np.random.choice(n_topics, n_samples)

    if return_counts:
        topic_range, document_range = np.arange(n_topics), np.arange(n_documents)
        topic_counts = (topics == topic_range[None].T)@np.eye(n_words)[words]
        document_counts = (documents == document_range[None].T)@np.eye(n_topics)[topics]

        return topics, topic_counts, document_counts
    
    return topics

class GibbsLDA(BaseEstimator, ClusterMixin, TransformerMixin):
    """Adaptation of latent Dirichlat allocation for point cloud clustering
    using collapsed Gibbs sampling. Based on the model proposed by David Blei, 
    et al. 

    https://papers.nips.cc/paper/2070-latent-dirichlet-allocation

    Parameters
    ----------
    n_topics : int, default=5
        Number of discoverable topics.
    n_words : int, default=50
        Number of phenotypic words.
    topic_prior : float, default=1.0
        Topic distribution Dirichlet prior.
    document_prior : float, default=1.0
        Document distribution Dirichlet prior.
    seed : int, default=None
        Random state seed.

    Attributes
    ----------
    corpus : ndarray
        Assignment values for each sample.
    topics : ndarray
        Topic assignment history for each sample.
    topic_counts : ndarray
        Word counts for each topic.
    document_counts : ndarray
        Topic counts for each document.
    likelihood_log : list
        Record of the total likelihood of each update step.
    labels_ : ndarray
        Final topic assignments for each sample.

    Usage
    -----
    >>> model = GibbsLDA(*args, **kwargs)
    >>> labels = model.fit_predict(data, **kwargs)
    >>> corpus = model.fit_transform(data, **kwargs)
    """
    
    def __init__(self, n_topics=5, n_words=50, topic_prior=1., document_prior=1., seed=None):
        super().__init__()
        set_seed(seed)

        self.n_topics = n_topics
        self.n_words = n_words
        self.topic_prior = topic_prior
        self.document_prior = document_prior
        self.seed = seed

        self.corpus = None
        self.topics = None
        self.topic_counts = None
        self.document_counts = None
        self.likelihood_log = []
        self.labels_ = None

    def _shuffle(self, documents, words):
        return shuffle(documents, words, self.n_topics, return_counts=True)
    
    def build(self, data, n_steps=200):
        """Initializes model parameters and class attributes.

        Parameters
        ----------
        data : ndarray
            Section and feature values for each sample.
        n_steps : int, default=200
            Number of Gibbs sampling steps.

        Returns
        -------
        self
            I return therefore I am.
        """
        
        documents, features = data[:, 0].astype(np.int32), data[:, 1:]
        words = KMeans(self.n_words).fit_predict(features)
        topics, self.topic_counts, self.document_counts = self._shuffle(documents, words)
        self.corpus = np.vstack([topics, documents, words]).T
        self.topics = np.zeros((n_steps, data.shape[0]))
        self.topics[-1:] = topics

        return self
    
    def decrement(self, topic, word):
        """Decrements the specified word count for the given topic.
        
        Parameters
        ----------
        topic : int
            Topic assignment.
        word : int
            Word assignment.

        Returns
        -------
        ndarray
            Updated word counts for each topic.
        """
        
        self.topic_counts[topic, word] -= 1

        return self.topic_counts
    
    def increment(self, topic, word):
        """Increments the specified word count for the given topic.
        
        Parameters
        ----------
        topic : int
            Topic assignment.
        word : int
            Word assignment.

        Returns
        -------
        ndarray
            Updated word counts for each topic.
        """

        self.topic_counts[topic, word] += 1

        return self.topic_counts
    
    def sample_topic(self, document, word, return_likelihood=False):
        """Samples a topic assignment based on the given document and word
        assignments.
        
        Parameters
        ----------
        document : int
            Document assignment.
        word : int
            Word assignment.
        return_likelihood : bool, default=False
            Whether to return the assignment likelihood.

        Returns
        -------
        int
            Topic assignment.
        float
            Assignment likelihood.
        """

        topic_distribution = self.document_counts[document] + self.document_prior
        topic_distribution /= topic_distribution.sum()
        word_distribution = self.topic_counts[:, word] + self.topic_prior
        word_distribution /= (self.topic_counts + self.topic_prior).sum(-1)
        distribution = topic_distribution*word_distribution
        distribution /= distribution.sum()
        topic = np.random.choice(self.n_topics, p=distribution)

        if return_likelihood:
            return topic, distribution[topic]
        
        return topic
    
    def update(self, step, sample):
        """Updates the given sample's topic assignment based on the current
        assignments.

        Parameters
        ----------
        step : int
            Gibbs sampling step.
        sample : int
            Sample index.

        Returns
        -------
        float
            Assignment likelihood.
        """
        
        topic, document, word = self.corpus[sample]
        self.decrement(topic, word)
        topic, likelihood = self.sample_topic(document, word, return_likelihood=True)
        self.increment(topic, word)
        self.corpus[sample, 0] = topic
        self.topics[step, sample] = topic

        return likelihood
    
    def step(self, step):
        """Performs a Gibbs sampling update for each sample and returns the 
        total step likelihood.
        
        Parameters
        ----------
        step : int
            Gibbs sampling step.

        Returns
        -------
        float
            Total step likelihood.
        """

        samples = np.random.permutation(self.corpus.shape[0])
        likelihood = 0

        for s in samples:
            likelihood += self.update(step, s)

        return likelihood
    
    def fit(self, data, n_steps=200, burn_in=150, description='LDA', verbosity=1):
        """Computes a topic assignment for each sample using collapsed Gibbs 
        sampling.
        
        Parameters
        ----------
        data : ndarray
            Section and feature values for each sample.
        n_steps : int, default=200
            Number of Gibbs sampling steps.
        burn_in : int, default=150
            Number of sampling steps to discard.
        description : str, default='LDA'
            Model description.
        verbosity : int, default=1
            Level of information logging.

        Returns
        -------
        self
            I return therefore I am.
        """
        
        self.build(data, n_steps)

        for i in tqdm(range(n_steps), description) if verbosity == 1 else range(n_steps):
            likelihood = self.step(i)
            self.likelihood_log.append(likelihood)

        self.labels_, _ = stats.mode(self.topics[burn_in:], 0)

    def transform(self, _=None):
        """Returns the assignment values for each sample.
        
        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Assignment values for each sample.
        """
        
        return self.corpus
    
# def shuffle(words, documents, n_topics=5, return_counts=False):
#     n_samples, n_documents = words.shape[0], np.unique(documents).shape[0]
#     topic_assignments = np.random.choice(n_topics, (n_samples, 1))

#     if return_counts:
#         topic_counts = np.zeros((n_topics, np.unique(words).shape[0]))
        
#         for i in range(n_topics):
#             w = words[topic_assignments[:, 0] == i]
#             u, c = np.unique(w, return_counts=True)
#             topic_counts[i, u] = c

#         document_counts = np.zeros((n_documents, n_topics))

#         for i in range(n_documents):
#             t = topic_assignments[documents == i]
#             u, c = np.unique(t, return_counts=True)
#             document_counts[i, u] = c

#         return topic_assignments, document_counts, topic_counts
    
#     return topic_assignments

# class LDA(BaseEstimator, TransformerMixin, ClusterMixin):
#     def __init__(self, n_topics=5, n_words=50, document_prior=1., topic_prior=1., seed=None):
#         super().__init__()
#         set_seed(seed)

#         self.n_topics = n_topics
#         self.n_words = n_words
#         self.document_prior = document_prior
#         self.topic_prior = topic_prior
#         self.seed = seed

#         self.documents = None
#         self.topics = None
#         self.corpus = None
#         self.document_counts = None
#         self.topic_counts = None
#         self.likelihood_log = []
#         self.burn_in = 150
#         self.labels_ = None

#     def _shuffle(self, words, documents):
#         return shuffle(words, documents, self.n_topics, return_counts=True)

#     def build(self, data, n_steps, burn_in):
#         _, bins = np.histogram(data, self.n_words)
#         self.documents = np.digitize(data, bins)
#         (n_documents, n_words), n_samples = self.documents.shape, self.documents.sum()
#         documents = np.arange(n_documents).repeat(self.documents.sum(-1))
#         words = np.hstack([np.arange(n_words).repeat(self.documents[i]) for i in range(n_documents)])
#         topics, self.document_counts, self.topic_counts = self._shuffle(words, documents)
#         self.corpus = np.hstack([words[None].T, documents[None].T, topics])
#         self.topics = np.zeros((n_steps, n_samples))
#         self.topics[-1:] = topics.T
#         self.burn_in = burn_in

#         return self
    
#     def decrement(self, word, topic):
#         self.topic_counts[topic, word] -= 1

#         return self.topic_counts
    
#     def increment(self, word, topic):
#         self.topic_counts[topic, word] += 1

#         return self.topic_counts
    
#     def sample(self, word, document, return_likelihood=False):
#         topic_distribution = self.document_counts[document] + self.document_prior
#         topic_distribution /= (self.document_counts[document] + self.document_prior).sum()
#         word_distribution = self.topic_counts[:, word] + self.topic_prior
#         word_distribution /= (self.topic_counts + self.topic_prior).sum(-1)
#         distribution = topic_distribution*word_distribution
#         distribution /= distribution.sum()
#         topic = np.random.choice(self.n_topics, p=distribution)

#         if return_likelihood:
#             return topic, distribution[topic]
        
#         return topic
    
#     def update(self, step_n, sample):
#         word, document, topic = self.corpus[sample]
#         self.decrement(word, topic)
#         topic, likelihood = self.sample(word, document, return_likelihood=True)
#         self.increment(word, topic)
#         self.corpus[sample, -1] = topic
#         self.topics[step_n, sample] = topic

#         return likelihood
    
#     def step(self, step_n):
#         n_samples = self.corpus.shape[0]
#         samples = np.random.permutation(n_samples)
#         likelihood = 0

#         for s in samples:
#             likelihood += self.update(step_n, s)

#         return likelihood

#     def fit(self, data, n_steps=200, burn_in=150, description='LDA', verbosity=1):
#         self.build(data, n_steps, burn_in)

#         for i in tqdm(range(n_steps), description) if verbosity == 1 else range(n_steps):
#             likelihood = self.step(i)
#             self.likelihood_log.append(likelihood)

#         self.labels_, _ = stats.mode(self.topics[self.burn_in:], 0)

#         return self
