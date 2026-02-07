'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from functools import singledispatch, wraps
from inspect import getcallargs
from ._utils import check_data, get_kwargs

__all__ = [
    'attrmethod',
    'buildmethod',
    'checkmethod'
]

@singledispatch
def attrmethod(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        method_kwargs = dict(getcallargs(method, self, *args), **kwargs)
        del method_kwargs['self']

        for key, val in method_kwargs.items():
            if val is not None or not hasattr(self, key):
                setattr(self, key, val)

        return method(self, *args, **kwargs)
    return wrapper

@attrmethod.register(str)
def _(prefix='', suffix=''):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            method_kwargs = dict(getcallargs(method, self, *args), **kwargs)
            del method_kwargs['self']

            for key, val in method_kwargs.items():
                key = prefix + key + suffix

                if val is not None or not hasattr(self, key):
                    setattr(self, key, val)

            return method(self, *args, **kwargs)
        return wrapper
    return decorator

@singledispatch
def buildmethod(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, '_build'):
            method_kwargs = dict(getcallargs(method, self, *args), **kwargs)
            build_kwargs = get_kwargs(self._build, **method_kwargs)
            self._build(**build_kwargs)

        return method(self, *args, **kwargs)
    return wrapper

@buildmethod.register(str)
def _(*builders):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            for builder in builders:
                if hasattr(self, builder):
                    build = getattr(self, builder)
                    method_kwargs = dict(getcallargs(method, self, *args), **kwargs)
                    build_kwargs = get_kwargs(build, **method_kwargs)
                    build(**build_kwargs)

            return method(self, *args, **kwargs)
        return wrapper
    return decorator

def checkmethod(method, accept_complex=False, accept_sparse=False, accept_large_sparse=False, dtype='numeric', order=None, ensure_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, estimator=None, input_name=''):
    @wraps(method)
    def wrapper(self, X, *args, **kwargs):
        if not hasattr(self, 'check') or self.check:
            X = check_data(X,
                self.accept_complex if hasattr(self, 'accept_complex') else accept_complex,
                self.accept_sparse if hasattr(self, 'accept_sparse') else accept_sparse,
                self.accept_large_sparse if hasattr(self, 'accept_large_sparse') else accept_large_sparse,
                self.dtype if hasattr(self, 'dtype') else dtype,
                self.order if hasattr(self, 'order') else order,
                self.ensure_all_finite if hasattr(self, 'ensure_all_finite') else ensure_all_finite,
                self.ensure_2d if hasattr(self, 'ensure_2d') else ensure_2d,
                self.allow_nd if hasattr(self, 'allow_nd') else allow_nd,
                self.ensure_min_samples if hasattr(self, 'ensure_min_samples') else ensure_min_samples,
                self.ensure_min_features if hasattr(self, 'ensure_min_features') else ensure_min_features,
                self.estimator if hasattr(self, 'estimator') else estimator,
                self.input_name if hasattr(self, 'input_name') else input_name
            )

        return method(self, X, *args, **kwargs)
    return wrapper
