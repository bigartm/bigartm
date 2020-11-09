from __future__ import print_function

from setuptools import setup, find_packages, Distribution
from setuptools.command.build_py import build_py
from distutils.spawn import find_executable


from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

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




# name of artm shared library
artm_library_name = 'libartm.so'
if sys.platform.startswith('win'):
    artm_library_name = 'artm.dll'
elif sys.platform.startswith('darwin'):
    artm_library_name = 'libartm.dylib'

path_to_lib = src_abspath + 'python/artm/wrapper/' + artm_library_name

# setuptools to CMake solution based on https://github.com/pybind/cmake_example/
# which is based on https://github.com/YannickJadoul/Parselmouth/blob/master/setup.py


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            # TODO: check CMake version
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        print(f"running cmake from {extdir}")
        if not os.path.exists(extdir):
            os.makedirs(extdir)

        cmake_process = [cmake_exec]
        cmake_process.append(ext.sourcedir)
        if os.environ.get("AUDITWHEEL_PLAT") or os.environ.get("CIBW_BUILD"):
            cmake_process.append("-DBUILD_PIP_DIST=ON")
        else:
            cmake_process.append("-DBUILD_PIP_DIST=OFF")
        cmake_process.append('-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir)
        # cmake_process.append('-DPYTHON_EXECUTABLE=' + sys.executable)

        subprocess.check_call(cmake_process, cwd=extdir)

        # dirty hack to fix librt issue
        if os.environ.get("AUDITWHEEL_PLAT"):
            link_path = extdir + "/src/artm/CMakeFiles/artm.dir/link.txt"
            with open(link_path, "r") as link:
                contents = link.read().strip()
            with open(link_path, "w") as link:
                link.write(contents + " -lrt" + "\n")

        # some debug info for CIBW
        if os.environ.get("AUDITWHEEL_PLAT") or os.environ.get("CIBW_BUILD"):
            result = subprocess.run(["ls"], stdout=subprocess.PIPE, cwd=extdir)
            warnings.warn(result.stdout.decode("utf8"))

            result = subprocess.run(["ls"], stdout=subprocess.PIPE, cwd=extdir + "/python/")
            warnings.warn(result.stdout.decode("utf8"))

        print(f"running make from {extdir}")
        make_process = ["make"]
        # make_process.append("-j6")
        subprocess.check_call(make_process, cwd=extdir)

        # removing extraneous artifacts
        if os.environ.get("AUDITWHEEL_PLAT") or os.environ.get("CIBW_BUILD"):
            print("cleaning up...")
            for bad_dir in ['3rdparty', 'CMakeFiles', 'src', 'bin', 'lib', 'python']:
                shutil.rmtree(extdir + "/" + bad_dir)

            for bad_file in ['CTestTestfile.cmake', 'cmake_install.cmake', 'CMakeCache.txt', 'Makefile']:
                os.remove(extdir + "/" + bad_file)

        # hack: copy libartm into /artm/wrapper/, where it belongs
        # instead of leaving it in the root direcetory where it mysteriously appeared
        shutil.move(
            extdir + "/" + artm_library_name,
            extdir + '/artm/wrapper/' + artm_library_name
        )
        #TODO: this message is linux-specific
        print(f"Now consider running\n"
              f"export ARTM_SHARED_LIBRARY={extdir}/libartm.so\n"
              f"so python is able to find compiled binary"
        )



class BinaryDistribution(Distribution):
    """
    This inheritor forces setuptools to include the "built" shared library into
    the binary distribution.
    """
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


with open(os.path.join(src_abspath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    package_data={'artm.wrapper': [path_to_lib]},
    include_package_data=True,
    packages=find_packages('./python/'),
    package_dir={'': './python/'},

    ext_modules=[CMakeExtension('bigartm')],
    cmdclass=dict(build_ext=CMakeBuild),

    long_description_content_type='text/markdown',
    long_description=long_description,
)
