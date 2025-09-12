'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from tqdm import tqdm
from ...utils import relabel

def distribute(locs, n_documents=None, scale=1., n_neighbors=4):
    """Uniformly distributes document locations proximally to sample locations
    and computes a local density-based variance for each document.
    
    Parameters
    ----------
    data : ndarray
        Sample dataset.
    n_documents : int, default=None
        Number of documents per section.
    scale : float, default=1.0
        Variance scale factor.
    n_neighbors : int, default=4
        Size of local neighborhood.

    Returns
    -------
    ndarray
        Document data.
    """

    sections = np.unique(locs[:, 0])
    documents = []

    for s in sections:
        mask = locs[:, 0] == s

        if n_documents is None:
            n_documents = mask.sum()//4

        idx = np.random.permutation(mask.sum())[:n_documents]
        doc_locs = locs[mask, :3][idx]
        proximity = cdist(doc_locs[:, 1:], doc_locs[:, 1:], 'sqeuclidean')
        variance = scale*np.sort(proximity, -1)[:, n_neighbors]
        documents.append(np.hstack([doc_locs, variance[None].T]))

    documents = np.vstack(documents)

    # sections = np.unique(data[:, 0])
    # documents = []

    # for s in sections:
    #     section = data[data[:, 0] == s]
    #     n_samples = section.shape[0]

    #     if n_documents is None:
    #         n_documents = n_samples//4

    #     idx = np.random.permutation(n_samples)[:n_documents]
    #     locations = section[idx, :3]
    #     proximity = cdist(locations[:, 1:], locations[:, 1:], 'sqeuclidean')
    #     scale = np.sort(proximity, -1)[:, 1:5].mean(-1)
    #     documents.append(np.hstack([locations, 4*scale[None].T]))

    # documents = np.vstack(documents)

    return documents

def featurize(data, locs, scale=1., n_neighbors=4):
    """Creates spatially-smoothed features by applying two Gaussian filters with
    local density-based variances to the given raw data.
    
    Parameters
    ----------
    data : ndarray
        Sample dataset.
    scale : float, default=1.0
        Smoothing scale factor.
    n_neighbors : int, default=4
        Size of local neighborhood.

    Returns
    -------
    ndarray
        Smoothed sample features. 
    """

    sections = np.unique(locs[:, 0])
    features = []

    for s in sections:
        mask = locs[:, 0] == s
        proximity = cdist(locs[mask, 1:], locs[mask, 1:], 'sqeuclidean')
        variance = scale*np.sort(proximity, -1)[:, n_neighbors]
        gaussian = np.exp(-proximity/(2*variance))/(np.sqrt(2*np.pi*variance))
        features.append(gaussian@data[mask])

    features = np.vstack(features)

    # sections = np.unique(data[:, 0])
    # features = []

    # for s in sections:
    #     section = data[data[:, 0] == s]
    #     proximity = cdist(section[:, 1:3], section[:, 1:3], 'sqeuclidean')
    #     scale = np.sort(proximity, -1)[:, 1:5].mean(-1)
    #     gaussian1 = np.exp(-proximity/(2*(.5*scale)**2))/np.sqrt(2*np.pi*(.5*scale)**2)
    #     gaussian2 = np.exp(-proximity/(2*(1.*scale)**2))/np.sqrt(2*np.pi*(1.*scale)**2)
    #     features.append(np.hstack([gaussian1@section[:, 3:], gaussian2@section[:, 3:]]))

    # features = np.vstack(features)

    return features

def shuffle(words, n_topics=5, n_documents=200, n_words=40, return_counts=False):
    """Randomly assigns a topic and document to each sample.
        
    Parameters
    ----------
    words : ndarray
        Word assignment for each sample
    n_topics : int, default=5
        Number of discoverable topics.
    n_documents : int, default=200
        Number of spatial documents.
    n_words : int, default=40
        Number of phenotypic words.
    return_counts : bool, default=False
        Whether to return assignment counts.

    Returns
    -------
    ndarray
        Topic assignments.
    ndarray
        Document assignments.
    ndarray
        Topic-word counts.
    ndarray
        Document-topic counts.
    """

    n_samples = words.shape[0]
    topics = np.random.choice(n_topics, n_samples)
    documents = np.random.choice(n_documents, n_samples)

    if return_counts:
        topic_range, document_range = np.arange(n_topics), np.arange(n_documents)
        topic_counts = (topics == topic_range[None].T)@np.eye(n_words)[words]
        document_counts = (documents == document_range[None].T)@np.eye(n_topics)[topics]

        return topics, documents, topic_counts, document_counts
    
    return topics, documents

class GibbsSLDA(BaseEstimator, ClusterMixin, TransformerMixin):
    """Adaptation of spatial latent Dirichlet allocation for point cloud 
    clustering using collapsed Gibbs sampling. Based on methods proposed by 
    Xiaogang Wang and Eric Grimson.

    https://papers.nips.cc/paper/3278-spatial-latent-dirichlet-allocation

    Parameters
    ----------
    n_topics : int, default=5
        Number of discoverable topics.
    n_documents : int | tuple | list, default=None
        Number of documents per section.
    n_words : int, default=40
        Number of phenotypic words.
    document_scale : float, default=1.0
        Document variance scale factor.
    word_scale : float, default=1.0
        Word smoothing scale factor.
    topic_prior : float, default=1.0
        Topic distribution Dirichlet prior.
    document_prior : float, default=1.0
        Document distribution Dirichlet prior.
    seed : int, default=None
        Random state seed.

    Attributes
    ----------
    corpus : ndarray
        Section, coordinate, and assignment values for each sample.
    topics : ndarray
        Topic assignment history for each sample.
    documents : ndarray
        Section, coordinate, and variance values for each document.
    topic_counts : ndarray
        Word counts for each topic.
    document_counts : ndarray
        Topic counts for each document.
    likelihood_log : list
        Record of the total likelihood for each update step.
    labels_ : ndarray
        Final topic assignments for each sample.
    
    Usage
    -----
    >>> model = GibbsSLDA(*args, **kwargs)
    >>> predictions = model.fit_predict(data, **kwargs)
    >>> corpus = model.fit_transform(data, **kwargs)
    """

    def __init__(self, n_topics=3, n_documents=None, n_words=32, document_scale=1., word_scale=1., topic_prior=1., document_prior=1.):
        super().__init__()

        self.n_topics = n_topics
        self.n_documents = n_documents
        self.n_words = n_words
        self.document_scale = document_scale
        self.word_scale = word_scale
        self.topic_prior = topic_prior
        self.document_prior = document_prior

        self.corpus = None
        self.topics = None
        self.documents = None
        self.topic_counts = None
        self.document_counts = None
        self.likelihood_log = []
        self.labels_ = None

    def _distribute(self, locs):
        return distribute(locs, self.n_documents, self.document_scale)

    def _featurize(self, data, locs):
        return featurize(data, locs, self.word_scale)
    
    def _shuffle(self, words):
        return shuffle(words, self.n_topics, self.documents.shape[0], self.n_words, return_counts=True)
    
    def build(self, data, locs, n_steps=400):
        """Initializes model parameters and class attributes.
        
        Parameters
        ----------
        data : ndarray
            Section, coordinate, and feature values for each sample.
        n_steps : int, default=400
            Number of Gibbs sampling steps.

        Returns
        -------
        self
            I return therefore I am.
        """

        self.documents, features = self._distribute(locs), self._featurize(data, locs)
        words = KMeans(self.n_words).fit_predict(features)
        topics, documents, self.topic_counts, self.document_counts = self._shuffle(words)
        self.corpus = np.vstack([locs.T, topics, documents, words]).T
        self.topics = np.zeros((n_steps, data.shape[0]))
        self.topics[-1:] = topics

        return self
    
    def decrement(self, topic, document, word):
        """Decrements the specified word and topic counts for the given topic
        and document, respectively.
        
        Parameters
        ----------
        topic : int
            Topic assignment.
        document : int
            Document assignment.
        word : int
            Word value.

        Returns
        -------
        ndarray
            Updated word counts for each topic.
        ndarray
            Updated topic counts for each document.
        """

        self.topic_counts[topic, word] -= 1
        self.document_counts[document, topic] -= 1

        return self.topic_counts, self.document_counts
    
    def increment(self, topic, document, word):
        """Increments the specified word and topic counts for the given topic
        and document, respectively.
                
        Parameters
        ----------
        topic : int
            Topic assignment.
        document : int
            Document assignment.
        word : int
            Word value.

        Returns
        -------
        ndarray
            Updated word counts for each topic.
        ndarray
            Updated topic counts for each document.
        """

        self.topic_counts[topic, word] += 1
        self.document_counts[document, topic] += 1

        return self.topic_counts, self.document_counts
    
    def sample_topic(self, document, word, return_likelihood=False):
        """Samples a topic assignment based on the given document and word
        assignments.
        
        Parameters
        ----------
        document : int
            Document assignment.
        word : int
            Word value.
        return_likelihood : bool, default=False
            Whether to return the assignment likelihood.

        Returns
        -------
        int
            Topic assignment.
        ndarray
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
    
    def sample_document(self, location, topic, return_likelihood=False):
        """Samples a document assignment based on the given spatial location and 
        topic assignment.
        
        Parameters
        ----------
        location : ndarray
            Section and coordinate values for a sample.
        topic : int
            Topic assignment.
        return_likelihood : bool, default=False
            Whether to return the assignment likelihood.

        Returns
        -------
        int
            Document assignment.
        ndarray
            Assignment likelihood.
        """

        mask = self.documents[:, 0] == location[0]
        proximity = cdist(location[1:][None], self.documents[mask, 1:3], 'sqeuclidean')[0]
        document_distribution = np.exp(-proximity/self.documents[mask, -1])
        topic_distribution = self.document_counts[mask, topic] + self.document_prior
        topic_distribution /= (self.document_counts[mask] + self.document_prior).sum(-1)
        distribution = document_distribution*topic_distribution
        distribution /= distribution.sum()
        document = np.random.choice(mask.sum(), p=distribution)

        if return_likelihood:
            return document, distribution[document]
        
        return document
    
    def update(self, step, sample):
        """Updates the given sample's topic and document assignments based on
        the current assignments.

        Parameters
        ----------
        step_n : int
            Gibbs sampling step.
        sample : int
            Sample index.

        Returns
        -------
        float
            Assignment likelihood.
        """

        location, (topic, document, word) = self.corpus[sample, :3], self.corpus[sample, 3:].astype(np.int32)
        self.decrement(topic, document, word)
        new_topic, topic_likelihood = self.sample_topic(document, word, return_likelihood=True)
        new_document, document_likelihood = self.sample_document(location, topic, return_likelihood=True)
        likelihood = topic_likelihood + document_likelihood
        self.increment(new_topic, new_document, word)
        self.corpus[sample, 3], self.corpus[sample, 4] = new_topic, new_document
        self.topics[step, sample] = new_topic

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
    
    def fit(self, data, locs=None, labels=None, n_steps=400, burn_in=300, description='SLDA', verbosity=1):
        """Computes topic and document assignments for each sample using
        collapsed Gibbs sampling.

        Parameters
        ----------
        data : ndarray
            Section, coordinate, and feature values for each sample.
        n_steps : int, default=400
            Number of Gibbs sampling steps.
        burn_in : int, default=300
            Number of sampling steps to discard.
        description : str, default='SLDA'
            Model description.
        verbosity : int, default=1
            Level of information logging.

        Returns
        -------
        self
            I return therefore I am.
        """

        if locs is None:
            data, locs = data[:, 3:], data[:, :3]

        self.build(data, locs, n_steps)

        for i in tqdm(range(n_steps), description) if verbosity == 1 else range(n_steps):
            likelihood = self.step(i)
            self.likelihood_log.append(likelihood)

        self.labels_ = relabel(stats.mode(self.topics[burn_in:], 0).mode, labels)

        return self
    
    def transform(self, _=None):
        """Returns the section, coordinate, topic, document, and word values for
        each sample.
        
        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Section, coordinate, and assignment values for each sample.
        """

        return self.corpus
