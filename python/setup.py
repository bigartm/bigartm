# -*- coding: utf-8 -*-
# This file is for developer's use

from __future__ import print_function

from setuptools import setup, find_packages
from distutils.spawn import find_executable

# DISTUTILS_DEBUG = True

# parse arguments
import sys
import os.path
import tempfile
import shutil
import subprocess
import argparse

# specify classifiers
BIGARTM_CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Software Development'
]

# name of artm shared library
artm_library_name = 'libartm.so'
if sys.platform.startswith('win'):
    artm_library_name = 'artm.dll'
elif sys.platform.startswith('darwin'):
    artm_library_name = 'libartm.dylib'

setup(
    # some common information
    name='bigartm',
    version='0.8.1rc4-r3',
    packages=find_packages(),
    package_data={'artm.wrapper': [artm_library_name]},

    # information about dependencies
    install_requires=[
        'pandas',
        'numpy'
    ],
    # this option must solve problem with installing
    # numpy as dependency during `setup.py install` execution
    # some explanations here:
    # https://github.com/nengo/nengo/issues/508#issuecomment-64962892
    # https://github.com/numpy/numpy/issues/2434#issuecomment-65252402
    setup_requires=[
        'numpy'
    ],

    # metadata for upload to PyPI
    license='New BSD license',
    url='https://github.com/bigartm/bigartm',
    description='BigARTM: the state-of-the-art platform for topic modeling',
    classifiers=BIGARTM_CLASSIFIERS,
    # Who should referred as author and how?
    # author = 'Somebody'
    # author_email = 'Somebody\'s email'
    # Now include `artm_dev` Google group as primary maintainer
    maintainer='ARTM developers group',
    maintainer_email='artm_dev+pypi_develop@googlegroups.com'
)
