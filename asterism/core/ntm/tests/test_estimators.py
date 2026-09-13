'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from sklearn.utils.estimator_checks import parametrize_with_checks
from .._ntm import *
from .._rsb import *

_ESTIMATORS = [
    NTM(mode='softmax'),
    NTM(mode='dirichlet'),
    RSB()
]

_EXPECTED_FAILED_CHECKS = lambda _: {
    'check_clustering': '',
    'check_dont_overwrite_parameters': '',
    'check_no_attributes_set_in_init': ''
}

@parametrize_with_checks(_ESTIMATORS, expected_failed_checks=_EXPECTED_FAILED_CHECKS, xfail_strict=False)
def test_estimators(estimator, check):
    check(estimator)
