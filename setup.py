'''
Authors: Craig Fouts
License: Apache 2.0 license
'''

import numpy as np
from setuptools import find_packages, setup, Extension

setup(
    name='asterism',
    version='0.0.1',
    description='Experiments with point cloud clustering.',
    author='Craig Fouts',
    author_email='foutscw@gmail.com',
    packages=find_packages(),
    ext_modules=[
        Extension('asterism.utils._utils_', ['.\\asterism\\utils\\_utils_.c'], include_dirs=[np.get_include()])
    ]
)
