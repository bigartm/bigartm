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


# Find the Protocol Buffer Compiler.
def find_protoc_exec():
    # extract path to protobuf executable from command-line arguments
    argument_prefix = "protoc_executable"
    parser = argparse.ArgumentParser(description="", add_help=False)
    parser.add_argument("--{}".format(argument_prefix), action="store")
    found_args, rest_args = parser.parse_known_args(sys.argv)
    sys.argv = rest_args
    result = vars(found_args).get(argument_prefix, None)
    if result is not None and os.path.exists(result):
        return result
    # try to guess from environment variables
    if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
        return os.environ['PROTOC']
    # try to find using distutils helper function
    return find_executable("protoc")


protoc_exec = find_protoc_exec()


def generate_proto_files(
        src_folder,
        src_proto_file,
        dst_py_file):
    """
    Generates pb2.py files from corresponding .proto files
    """

    source_file = os.path.join(src_folder, src_proto_file)
    output_file = dst_py_file

    if (not os.path.exists(output_file) or
            os.path.exists(output_file) and
            os.path.getmtime(source_file) > os.path.getmtime(output_file)):
        print("Generating {}...".format(dst_py_file))

        if not os.path.exists(source_file):
            sys.stderr.write("Can't find required file: {}\n".format(
                source_file))
            sys.exit(-1)

        if not protoc_exec:
            raise ValueError("No protobuf compiler executable was found!")

        try:
            tmp_dir = tempfile.mkdtemp(dir="./")
            protoc_command = [
                protoc_exec,
                "-I" + src_folder,
                "--python_out=" + tmp_dir,
                source_file]
            print("Executing {}...".format(protoc_command))
            if subprocess.call(protoc_command):
                raise
            src_py_file = src_proto_file.replace(".proto", "_pb2.py")
            if os.path.exists(dst_py_file):
                os.remove(dst_py_file)
            os.rename(os.path.join(tmp_dir, src_py_file), dst_py_file)
        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)


from distutils.command.build import build as _build


class build(_build):
    def run(self):
        # Generate necessary .proto file if it doesn't exist.
        generate_proto_files(
            "../src",
            "./artm/messages.proto",
            "./artm/wrapper/messages_pb2.py")

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
            return
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


setup_kwargs = dict(
    name='bigartm',
    version='0.10.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'packaging',
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
)

if sys.argv[1] == "bdist_wheel":
    # we only mess up with those hacks if we are building a wheel
    setup_kwargs['distclass'] = BinaryDistribution
    setup_kwargs['cmdclass']['build_py'] = AddLibraryBuild

setup(**setup_kwargs)

