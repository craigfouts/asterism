'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from sklearn.utils.estimator_checks import parametrize_with_checks
from .._lda import GibbsLDA

_ESTIMATORS = [
    GibbsLDA(n_topics=3, doc_size=8),
]

@parametrize_with_checks(_ESTIMATORS)
def test_estimators(estimator, check):
    check(estimator)
