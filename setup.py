'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from setuptools import find_packages, setup, Extension

setup(
    name='asterism',
    version='0.0.1',
    description='Experiments with point cloud clustering.',
    author='Craig Fouts',
    author_email='c.fouts25@imperial.ac.uk',
    packages=find_packages(),
    ext_modules=[
        Extension('asterism.utils._utils_', ['./asterism/utils/_utils_.c'])
    ]
)
