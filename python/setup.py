from __future__ import print_function

from setuptools import setup, find_packages
from distutils.spawn import find_executable

# parse arguments
import sys
import os.path
import tempfile
import shutil
import subprocess

# Find the Protocol Compiler.
if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
    protoc_exec = os.environ['PROTOC']
elif os.path.exists("../build/3rdparty/protobuf-cmake/protoc/protoc"):
    print("Find protoc in 3rdparty")
    protoc_exec = "../build/3rdparty/protobuf-cmake/protoc/protoc"
elif os.path.exists("../build/3rdparty/protobuf-cmake/protoc/protoc.exe"):
    protoc_exec = "../build/3rdparty/protobuf-cmake/protoc/protoc.exe"
else:
    protoc_exec = find_executable("protoc")

if not protoc_exec:
    raise ValueError("No protobuf compiler executable was found!")


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

        try:
            tmp_dir = tempfile.mkdtemp(dir="./")
            protoc_command = [
                protoc_exec,
                "-I" + src_folder,
                "--python_out=" + tmp_dir,
                source_file]
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


setup(
    name='bigartm',
    version='0.7.5',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy'
    ],
    cmdclass={'build': build},
)
