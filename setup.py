from __future__ import print_function

from setuptools import setup, find_packages, Distribution
from setuptools.command.build_py import build_py
from distutils.spawn import find_executable

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
    src_abspath = os.path.dirname(__file__)
else:
    raise ValueError("Cannot determine working directory!")

working_dir = os.path.abspath(os.getcwd())

# maybe we are inside Travis container? Fallback
src_abspath = os.environ.get('CI_BUILD_DIR')
if src_abspath is None:
    if os.environ.get("AUDITWHEEL_PLAT"):
        # we are inside PyPa manylinux docker
        src_abspath = "/project/"
    elif shutil.which("python") == "/tmp/cibw_bin/python":
        # we are inside macosx virtual machine
        username = 'bt2901'  # TODO: do not hardcode this
        src_abspath = f"/Users/travis/build/{username}/bigartm/"
    else:
        src_abspath = working_dir


setup_kwargs = {}

# name of artm shared library
artm_library_name = 'libartm.so'
if sys.platform.startswith('win'):
    artm_library_name = 'artm.dll'
elif sys.platform.startswith('darwin'):
    artm_library_name = 'libartm.dylib'

path_to_lib = src_abspath + 'python/artm/wrapper/' + artm_library_name


class build(_build):
    def run(self):
        try:
            warnings.warn('inside run()')
            build_directory = tempfile.mkdtemp(dir=src_abspath)
            # run cmake
            cmake_process = [cmake_exec]
            cmake_process.append(src_abspath)
            cmake_process.append("-DBUILD_PIP_DIST=ON")
            # FIXME
            # validate return code
            retval = subprocess.call(cmake_process, cwd=build_directory)
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
            retval = subprocess.call(make_process, cwd=build_directory)
            if retval:
                sys.exit(-1)
            # run make install command
            install_process = ["make", "install"]
            retval = subprocess.call(install_process, cwd=build_directory)
            if retval:
                sys.exit(-1)
            # result = subprocess.run(["ls"], stdout=subprocess.PIPE, cwd=build_directory)
            # warnings.warn(result.stdout.decode("utf8"))

            # result = subprocess.run(["ls"], stdout=subprocess.PIPE, cwd=src_abspath + '../../../../../../Users')
            # warnings.warn(result.stdout.decode("utf8"))
            # result = subprocess.run(["ls"], stdout=subprocess.PIPE, cwd=src_abspath + '../../../../../../')
            # warnings.warn(result.stdout.decode("utf8"))
        finally:
            if os.path.exists(build_directory):
                shutil.rmtree(build_directory)
        # _build is an old-style class, so super() doesn't work.
        _build.run(self)


class BinaryDistribution(Distribution):
    """
    This inheritor forces setuptools to include the "built" shared library into
    the binary distribution.
    """
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

if sys.argv[1] == "bdist_wheel":
    # we only mess up with those hacks if we are building a wheel
    setup_kwargs['distclass'] = BinaryDistribution
    setup_kwargs['cmdclass'] = {'build': build}

setup(
    package_data={'artm.wrapper': [path_to_lib]},
    include_package_data=True,
    packages=find_packages(src_abspath + 'python/'),
    package_dir={'': './python/'},

    **setup_kwargs
)
