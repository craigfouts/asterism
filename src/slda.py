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

def featurize(data):
    """Creates spatially-smoothed features by applying two Gaussian filters with
    local density-based variances to the given raw data.
    
    Parameters
    ----------
    data : ndarray
        Sample dataset.

    Returns
    -------
    ndarray
        Smoothed sample features.  
    """

    sections = np.unique(data[:, 0])
    features = []

    for s in sections:
        section = data[data[:, 0] == s]
        proximity = cdist(section[:, 1:3], section[:, 1:3], 'sqeuclidean')
        scale = np.sort(proximity, -1)[:, 1:5].mean(-1)
        gaussian1 = np.exp(-proximity/(2*(.5*scale)**2))/np.sqrt(2*np.pi*(.5*scale)**2)
        gaussian2 = np.exp(-proximity/(2*(1.*scale)**2))/np.sqrt(2*np.pi*(1.*scale)**2)
        features.append(np.hstack([gaussian1@section[:, 3:], gaussian2@section[:, 3:]]))

    features = np.vstack(features)

    # sections = np.unique(data[:, 0])
    # features = []

    # for s in sections:
    #     mask = data[:, 0] == s
    #     proximity = cdist(data[mask, 1:3], data[mask, 1:3], 'sqeuclidean')
    #     variance = scale*np.sort(proximity, -1)[:, n_neighbors]
    #     gaussian = np.exp(-proximity/(2*variance))/(np.sqrt(2*np.pi*variance))
    #     features.append(gaussian@data[mask, 3:])

    # features = np.vstack(features)

    return features

def distribute(data, n_documents=None):
    """Uniformly distributes document locations proximally to sample locations
    and computes a local density-based variance for each document.
    
    Parameters
    ----------
    data : ndarray
        Sample dataset.
    n_documents : int, default=None
        Number of documents per section.

    Returns
    -------
    ndarray
        Document data.
    """

    sections = np.unique(data[:, 0])
    documents = []

    for s in sections:
        section = data[data[:, 0] == s]
        n_samples = section.shape[0]

        if n_documents is None:
            n_documents = n_samples//4

        idx = np.random.permutation(n_samples)[:n_documents]
        locations = section[idx, :3]
        proximity = cdist(locations[:, 1:], locations[:, 1:], 'sqeuclidean')
        scale = np.sort(proximity, -1)[:, 1:5].mean(-1)
        documents.append(np.hstack([locations, 4*scale[None].T]))

    documents = np.vstack(documents)

    # sections = np.unique(data[:, 0])
    # documents = []

    # for s in sections:
    #     mask = data[:, 0] == s

    #     if n_documents is None:
    #         n_documents = mask.sum()//4

    #     idx = np.random.permutation(mask.sum())[:n_documents]
    #     locs = data[mask, :3][idx]
    #     proximity = cdist(locs[:, 1:], locs[:, 1:], 'sqeuclidean')
    #     variance = scale*np.sort(proximity, -1)[:, n_neighbors]
    #     documents.append(np.hstack([locs, variance[None].T]))

    # documents = np.vstack(documents)

    return documents

def shuffle(words, documents, n_topics=5, n_words=50, return_counts=False):
    """Randomly initializes document and topic assignments for each sample.
    
    Parameters
    ----------
    words : ndarray
        Word assignment for each sample
    documents : ndarray
        Document data.
    n_topics : int, default=5
        Number of discoverable topics.
    n_words : int, default=50
        Number of phenotypic words.
    return_counts : bool, default=False
        Whether to return assignment counts.

    Returns
    -------
    ndarray
        Document assignments.
    ndarray
        Topic assignments.
    ndarray
        Document-topic counts.
    ndarray
        Topic-word counts.
    """

    n_samples, n_documents = words.shape[0], documents.shape[0]
    document_assignments = np.random.choice(n_documents, (n_samples, 1))
    topic_assignments = np.random.choice(n_topics, (n_samples, 1))

    if return_counts:
        doc_range, topic_range = np.arange(n_documents), np.arange(n_topics)
        doc_counts = (document_assignments == doc_range).T@np.eye(n_topics)[topic_assignments.T[0]]
        topic_counts = (topic_assignments == topic_range).T@np.eye(n_words)[words]

        return document_assignments, topic_assignments, doc_counts, topic_counts
    
    return document_assignments, topic_assignments

class SLDA(BaseEstimator, ClusterMixin, TransformerMixin):
    """Adaptation of spatial latent Dirichlet allocation for spatial point
    cloud clustering. Based on the model proposed by Xiaogang Wang and Eric
    Grimson.

    https://papers.nips.cc/paper/3278-spatial-latent-dirichlet-allocation

    Parameters
    ----------
    n_topics : int, default=5
        Number of discoverable topics.
    n_documents : int | tuple | list, default=None
        Number of documents per section.
    n_words : int, default=50
        Number of phenotypic words.
    feature_scale : float, default=1.0
        Feature smoothing scale factor.
    document_scale : float, default=1.0
        Document variance scale factor.
    document_prior : float | ndarray, default=1.0
        Document-topic distribution Dirichlet prior.
    topic_prior : float | ndarray, default=1.0
        Topic-word distribution Dirichlet prior.
    seed : int, default=None
        Randome state seed.

    Attributes
    ----------
    documents : ndarray
        Location and variance of each document.
    corpus : ndarray
        Location and parameterization of each sample.
        (section index, x-coordinate, y-coordinate, word assignment, document assignment, topic assignment)
    topics : ndarray
        Topic assignment history of each sample.
    document_counts : ndarray
        Document-topic counts.
    topic_counts : ndarray
        Topic-word counts.
    likelihood_log : list
        Record of the total likelihood computed at each step.
    burn_in : int
        Number of sampling steps to discard.
    labels_ : 
        Final topic assignments of each sample.
    
    Usage
    -----
    >>> model = SLDA(*args, **kwargs)
    >>> topics = model.fit_predict(data, **kwargs)
    >>> corpus = model.fit_transform(data, **kwargs)
    """

    def __init__(self, n_topics=5, n_documents=None, n_words=25, feature_scale=1., document_scale=1., document_prior=1., topic_prior=1., seed=None):
        super().__init__()
        set_seed(seed)

        self.n_topics = n_topics
        self.n_documents = n_documents
        self.n_words = n_words
        self.feature_scale = feature_scale
        self.document_scale = document_scale
        self.document_prior = document_prior
        self.topic_prior = topic_prior
        self.seed = seed

        self.documents = None
        self.topics = None
        self.corpus = None

        # self.corpus = None
        # self.documents = None
        # self.topic_log = None

        self.document_counts = None
        self.topic_counts = None
        self.likelihood_log = []
        self.burn_in = 150
        self.labels_ = None

    def _featurize(self, data):
        return featurize(data, self.feature_scale)
    
    def _distribute(self, data):
        return distribute(data, self.n_documents, self.document_scale)
    
    def _shuffle(self, words):
        return shuffle(words, self.documents, self.n_topics, self.n_words, return_counts=True)
    
    def build(self, data, n_steps=200, burn_in=150):
        """Initializes model parameters and class attributes.
        
        Parameters
        ----------
        data : ndarray
            Sample dataset.
            (section index, x-coordinate, y-coordinate, features...)
        n_steps : int, default=200
            Number of Gibbs sampling steps.
        burn_in : int, default=150
            Number of sampling steps to discard.

        Returns
        -------
        self
            I return therefore I am.
        """

        n_samples, locations = data.shape[0], data[:, :3]
        features, self.documents = self._featurize(data), self._distribute(data)
        words = KMeans(self.n_words).fit_predict(features)
        documents, topics, self.document_counts, self.topic_counts = self._shuffle(words)
        self.corpus = np.hstack([locations, words[None].T, documents, topics])
        self.topics = np.zeros((n_steps, n_samples))
        self.topics[-1:] = topics.T
        self.burn_in = burn_in

        return self
    
    def decrement(self, word, document, topic):
        """Decrements the given document-topic and topic-word counts.

        Parameters
        ----------
        word : int
            Word value.
        document : int
            Document assignment.
        topic : int
            Topic assignment.

        Returns
        -------
        ndarray
            Decremented document-topic counts.
        ndarray
            Decremented topic-word counts.
        """

        self.document_counts[document, topic] -= 1
        self.topic_counts[topic, word] -= 1

        return self.document_counts, self.topic_counts
    
    def increment(self, word, document, topic):
        """Increments the given document-topic and topic-word counts.
        
        Parameters
        ----------
        word : int
            Word value.
        document : int
            Document assignment.
        topic : int
            Topic assignment.

        Returns
        -------
        ndarray
            Incremented document-topic counts.
        ndarray
            Incremented topic-word counts.
        """

        self.document_counts[document, topic] += 1
        self.topic_counts[topic, word] += 1

        return self.document_counts, self.topic_counts
    
    def sample_document(self, location, topic, return_distribution=False):
        """Samples a document assignment based on spatial location and topic
        assignment.
        
        Parameters
        ----------
        location : ndarray
            Section index and spatial coordinates.
        topic : int
            Topic assignment.
        return_distribution : bool, default=False
            Whether to return document probabilities.

        Returns
        -------
        int
            Document assignment.
        ndarray
            Document probabilities.
        """

        mask = self.documents[:, 0] == location[0]
        proximity = cdist(location[1:][None], self.documents[mask, 1:3], 'sqeuclidean')[0]
        document_distribution = np.exp(-proximity/self.documents[mask, -1])
        topic_distribution = self.document_counts[mask, topic] + self.document_prior
        topic_distribution /= (self.document_counts[mask] + self.document_prior).sum(-1)
        distribution = document_distribution*topic_distribution
        distribution /= distribution.sum()
        document = np.random.choice(mask.sum(), p=distribution)

        if return_distribution:
            return document, distribution
        
        return document
    
    def sample_topic(self, word, document, return_distribution=False):
        """Samples a topic assignment based on word value and document
        assignment.
        
        Parameters
        ----------
        word : int
            Word value.
        document : int
            Document assignment.
        return_distribution : bool, default=False
            Whether to return topic probabilities.

        Returns
        -------
        int
            Topic assignment
        ndarray
            Topic probabilities.
        """

        topic_distribution = self.document_counts[document] + self.document_prior
        topic_distribution /= (self.document_counts[document] + self.document_prior).sum()
        word_distribution = self.topic_counts[:, word] + self.topic_prior
        word_distribution /= (self.topic_counts + self.topic_prior).sum(-1)
        distribution = topic_distribution*word_distribution
        distribution /= distribution.sum()
        topic = np.random.choice(self.n_topics, p=distribution)

        if return_distribution:
            return topic, distribution
        
        return topic
    
    def sample(self, location, word, document, topic, return_likelihood=False):
        """Samples new document and topic assignments based on location and 
        current assignments.
        
        Parameters
        ----------
        location : ndarray
            Section index and spatial coordinates.
        word : int
            Word value.
        document : int
            Current document assignment.
        topic : int
            Current topic assignment.
        return_likelihood : bool, default=False
            Whether to return the assignment likelihood.

        Returns
        -------
        int
            New document assignment.
        int
            New topic assignment.
        float
            Assignment likelihood.
        """

        new_document = self.sample_document(location, topic, return_likelihood)
        new_topic = self.sample_topic(word, document, return_likelihood)

        if return_likelihood:
            likelihood = new_document[1][new_document[0]] + new_topic[1][new_topic[0]]

            return new_document[0], new_topic[0], likelihood
        
        return new_document, new_topic
    
    def update(self, step_n, sample):
        """Updates the given sample's document and topic assignments based on
        the previous step using Gibbs sampling.

        Parameters
        ----------
        step_n : int
            Gibbs sampling step number.
        sample : int
            Sample index.

        Returns
        -------
        float
            Assignment likelihood.
        """

        location, (word, document, topic) = self.corpus[sample, :3], self.corpus[sample, 3:].astype(np.int32)
        self.decrement(word, document, topic)
        document, topic, likelihood = self.sample(location, word, document, topic, return_likelihood=True)
        self.increment(word, document, topic)
        self.corpus[sample, -2], self.corpus[sample, -1] = document, topic
        self.topics[step_n, sample] = topic

        return likelihood
    
    def step(self, step_n):
        """Performs a Gibbs sampling update for each sample and returns the 
        total step likelihood.
        
        Parameters
        ----------
        step_n : int
            Gibbs sampling step number.

        Returns
        -------
        float
            Total step likelihood.
        """

        n_samples = self.corpus.shape[0]
        samples = np.random.permutation(n_samples)
        likelihood = 0

        for s in samples:
            likelihood += self.update(step_n, s)

        return likelihood
    
    def fit(self, data, n_steps=200, burn_in=150, description='SLDA', verbosity=1):
        """Obtains document and topic assignments for each sample based on its 
        spatial location and featurization.

        Parameters
        ----------
        data : ndarray
            Sample dataset.
        n_steps : int, default=200
            Number of Gibbs sampling steps.
        burn_in : int, default=150
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

        self.build(data, n_steps, burn_in)

        for i in tqdm(range(n_steps), description) if verbosity == 1 else range(n_steps):
            likelihood = self.step(i)
            self.likelihood_log.append(likelihood)

        self.labels_, _ = stats.mode(self.topics[self.burn_in:], 0)

        return self
    
    def transform(self, _=None):
        """Returns the location and parameterization of each sample.
        
        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Location and parameterization of each sample.
        """

        return self.corpus
