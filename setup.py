from __future__ import print_function

from setuptools import setup, find_packages, Distribution
from setuptools.command.build_py import build_py
from distutils.spawn import find_executable

from python.setup_data import setup_kwargs
# parse arguments
import sys
import os.path
import tempfile
import shutil
import subprocess
import argparse


import warnings

from distutils.command.build import build as _build


# Initialize necessary variables
# guess name of cmake executable
# FIXME: make cross-platform guessing
cmake_exec = "cmake"

# find absolute path of working directory
try:
    filename = __file__
except NameError:
    filename = sys.argv[0]
filename = os.path.abspath(filename)
if os.path.dirname(filename):
    os.chdir(os.path.dirname(filename))
working_dir = os.path.abspath(os.getcwd())

# maybe we are inside Travis container? Fallback
src_abspath = os.environ.get('CI_BUILD_DIR')
if src_abspath is None:
    if os.environ.get("AUDITWHEEL_PLAT"):
        print("wow such virtualenv wow")
        src_abspath = "/project/"
    elif shutil.which("python") == "/tmp/cibw_bin/python":
        print("wow such macos such travis wow")
        src_abspath = "/Users/travis/build/bt2901/bigartm/"
    else:
        src_abspath = working_dir


class build(_build):
    def run(self):
        try:
            warnings.warn('inside run()')
            build_directory = tempfile.mkdtemp(dir=src_abspath)
            os.chdir(build_directory)
            # run cmake
            cmake_process = [cmake_exec]
            cmake_process.append(src_abspath)
            cmake_process.append("-DBUILD_PIP_DIST=ON")
            # FIXME
            # validate return code
            retval = subprocess.call(cmake_process)
            if retval:
                sys.exit(-1)

            # dirty hack to fix librt issue
            if os.environ.get("AUDITWHEEL_PLAT"):
                link_path = build_directory + "/src/artm/CMakeFiles/artm.dir/link.txt"
                with open(link_path, "r") as link:
                    contents = link.read().strip()
                with open(link_path, "w") as link:
                    link.write(contents + " -lrt" + "\n")

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
            result = subprocess.run(["ls"])
            warnings.warn(result.stdout)
        finally:
            os.chdir(src_abspath)
            if os.path.exists(build_directory):
                shutil.rmtree(build_directory)
        # _build is an old-style class, so super() doesn't work.
        _build.run(self)


class AddLibraryBuild(build_py):
    """
    This hacky inheritor adds the shared library into the binary distribution.
    We pretend that we generated our library and place it into the temporary
    build directory.
    """
    def run(self):
        if not self.dry_run:
            self.copy_library()
        build_py.run(self)

    def get_outputs(self, *args, **kwargs):
        outputs = build_py.get_outputs(*args, **kwargs)
        outputs.extend(self._library_paths)
        return outputs

    def copy_library(self, builddir=None):
        self._library_paths = []
        library = os.getenv("ARTM_SHARED_LIBRARY", None)
        if library is None:
            raise ValueError()
        destdir = os.path.join(self.build_lib, 'artm')
        self.mkpath(destdir)
        dest = os.path.join(destdir, os.path.basename(library))
        shutil.copy(library, dest)
        self._library_paths = [dest]


class BinaryDistribution(Distribution):
    """
    This inheritor forces setuptools to include the "built" shared library into
    the binary distribution.
    """
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


# name of artm shared library
artm_library_name = 'libartm.so'
if sys.platform.startswith('win'):
    artm_library_name = 'artm.dll'
elif sys.platform.startswith('darwin'):
    artm_library_name = 'libartm.dylib'


if sys.argv[1] == "bdist_wheel":
    # we only mess up with those hacks if we are building a wheel
    setup_kwargs['distclass'] = BinaryDistribution
    setup_kwargs['cmdclass'] = {}
    setup_kwargs['cmdclass']['build'] = build
    setup_kwargs['cmdclass']['build_py'] = AddLibraryBuild

setup(
    package_data={'artm.wrapper': [artm_library_name]},
    include_package_data=True,
    packages=find_packages(src_abspath),
    **setup_kwargs
)
