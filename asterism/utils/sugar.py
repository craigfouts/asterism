'''
Authors: Craig Fouts
Contact: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from functools import singledispatch, wraps
from inspect import getcallargs
from ._utils import check_data, get_kwargs, get_methods

__all__ = [
    'attrmethod',
    'buildmethod',
    'checkmethod'
]

@singledispatch
def attrmethod(method):
    @wraps(method)
    def wrapper(cls, *args, **kwargs):
        method_kwargs = dict(getcallargs(method, cls, *args), **kwargs)
        del method_kwargs['self']

        for key, val in method_kwargs.items():
            if val is not None or not hasattr(cls, key):
                setattr(cls, key, val)

        return method(cls, *args, **kwargs)
    return wrapper

@attrmethod.register(str)
def _(prefix='', suffix=''):
    def decorator(method):
        @wraps(method)
        def wrapper(cls, *args, **kwargs):
            method_kwargs = dict(getcallargs(method, cls, *args, **kwargs), **kwargs)
            del method_kwargs['self']

            for key, val in method_kwargs.items():
                key = prefix + key + suffix

                if val is not None or not hasattr(cls, key):
                    setattr(cls, key, val)

            return method(cls, *args, **kwargs)
        return wrapper
    return decorator

@singledispatch
def buildmethod(method):
    @wraps(method)
    def wrapper(cls, *args, **kwargs):
        builders = get_methods(cls, '_build')

        for builder in filter(lambda x: x != method.__name__, builders):
            build = getattr(cls, builder)
            method_kwargs = dict(getcallargs(method, cls, *args, **kwargs), **kwargs)
            build_kwargs = get_kwargs(build, **method_kwargs)
            build(**build_kwargs)

        return method(cls, *args, **kwargs)
    return wrapper

@buildmethod.register(str)
def _(*builders):
    def decorator(method):
        @wraps(method)
        def wrapper(cls, *args, **kwargs):
            for builder in filter(lambda x: hasattr(cls, x), builders):
                build = getattr(cls, builder)
                method_kwargs = dict(getcallargs(method, cls, *args, **kwargs), **kwargs)
                build_kwargs = get_kwargs(build, **method_kwargs)
                build(**build_kwargs)

            return method(cls, *args, **kwargs)
        return wrapper
    return decorator

def checkmethod(method, accept_complex=False, accept_sparse=False, accept_large_sparse=False, dtype='numeric', order=None, ensure_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, estimator=None, input_name=''):
    @wraps(method)
    def wrapper(cls, X, *args, **kwargs):
        if not hasattr(cls, 'check') or cls.check:
            X = check_data(X,
                cls.accept_complex if hasattr(cls, 'accept_complex') else accept_complex,
                cls.accept_sparse if hasattr(cls, 'accept_sparse') else accept_sparse,
                cls.accept_large_sparse if hasattr(cls, 'accept_large_sparse') else accept_large_sparse,
                cls.dtype if hasattr(cls, 'dtype') else dtype,
                cls.order if hasattr(cls, 'order') else order,
                cls.ensure_all_finite if hasattr(cls, 'ensure_all_finite') else ensure_all_finite,
                cls.ensure_2d if hasattr(cls, 'ensure_2d') else ensure_2d,
                cls.allow_nd if hasattr(cls, 'allow_nd') else allow_nd,
                cls.ensure_min_samples if hasattr(cls, 'ensure_min_samples') else ensure_min_samples,
                cls.ensure_min_features if hasattr(cls, 'ensure_min_features') else ensure_min_features,
                cls.estimator if hasattr(cls, 'estimator') else estimator,
                cls.input_name if hasattr(cls, 'input_name') else input_name
            )

        return method(cls, X, *args, **kwargs)
    return wrapper
