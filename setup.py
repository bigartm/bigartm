# -*- coding: utf-8 -*-

from __future__ import print_function

from setuptools import setup, find_packages
from distutils.spawn import find_executable

DISTUTILS_DEBUG = True

# parse arguments
import sys
import os.path
import tempfile
import shutil
import subprocess

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


from distutils.command.build import build as _build


# Initialize necessary variables
# guess name of cmake executable
# FIXME: make cross-platform guessing
cmake_exec = "cmake"
# name of artm shared library
artm_library_name = 'libartm.so'
if sys.platform.startswith('win'):
    artm_library_name = 'artm.dll'
elif sys.platform.startswith('darwin'):
    artm_library_name = 'libartm.dylib'
# find absolute path of working directory
try:
    filename = __file__
except NameError:
    filename = sys.argv[0]
filename = os.path.abspath(filename)
if os.path.dirname(filename):
    os.chdir(os.path.dirname(filename))
working_dir = os.path.abspath(os.getcwd())


# Hook to distribute platform-dependent wheels if necessary
from setuptools.dist import Distribution


class BigARTMDistribution(Distribution):
    def is_pure(self):
        return false


class build(_build):
    def run(self):
        # Create build directories and run cmake
        try:
            build_directory = tempfile.mkdtemp(dir="./")
            os.chdir(build_directory)
            # run cmake
            cmake_process = [cmake_exec]
            cmake_process.append("../")
            cmake_process.append("-DBUILD_PIP_DIST=ON")
            # FIXME
            # validate return code
            retval = subprocess.call(cmake_process)
            if retval:
                sys.exit(-1)
            # run make command
            make_process = ["make"]
            # make_process.append("-j6")
            retval = subprocess.call(make_process)
            if retval:
                sys.exit(-1)
            # run make install command
            install_process = ["make", "install"]
            retval = subprocess.call(install_process)
            if retval:
                sys.exit(-1)
        finally:
            os.chdir(working_dir)
            if os.path.exists(build_directory):
                shutil.rmtree(build_directory)
        # _build is an old-style class, so super() doesn't work.
        _build.run(self)


setup(
    # some common information
    name='bigartm',
    version='0.8.1',
    packages=['artm', 'artm.wrapper'],
    package_dir={'': './python'},
    # add shared library to package
    package_data={'artm.wrapper': [artm_library_name]},
    distclass=BigARTMDistribution,

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
    cmdclass={'build': build},

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
