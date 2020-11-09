# This file is for developer's use

from setuptools import setup, find_packages

import sys

# name of artm shared library
artm_library_name = 'libartm.so'
if sys.platform.startswith('win'):
    artm_library_name = 'artm.dll'
elif sys.platform.startswith('darwin'):
    artm_library_name = 'libartm.dylib'

setup(
    name='bigartm',
    version='0.10.2.dev0',
    # add shared library to package
    package_data={'artm.wrapper': [artm_library_name]},
    packages=find_packages(),
    install_requires=['pytest', 'pytest-forked', 'scipy'],
)
