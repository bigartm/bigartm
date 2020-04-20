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

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # source_file = os.path.join(src_folder, src_proto_file)
    source_file = os.path.join(dir_path, src_folder, src_proto_file)
    src_folder_abs = os.path.join(dir_path, src_folder)
    dst_py_file_abs = os.path.join(dir_path, '..', dst_py_file)
    output_file = dst_py_file_abs

    if (not os.path.exists(output_file) or
            os.path.exists(output_file) and
            os.path.getmtime(source_file) > os.path.getmtime(output_file)):
        print("Generating {}...".format(dst_py_file))
        print("Generating {}...".format(dst_py_file_abs))

        sys.stderr.write("src_folder {} exists: {}\n".format(src_folder, os.path.isdir(src_folder)))
        sys.stderr.write("full path to me is {}, working directory is: {}\n".format(dir_path, os.getcwd()))

        print("----")
        print(subprocess.check_output('pwd', cwd=dir_path))
        print(subprocess.check_output('ls', cwd=dir_path))
        print("----")
        print(subprocess.check_output('pwd', cwd=os.getcwd()))
        print(subprocess.check_output('ls', cwd=os.getcwd()))
        print("====")
        print(subprocess.check_output('pwd', cwd=os.getcwd() + "/artm/"))
        print(subprocess.check_output('ls', cwd=os.getcwd() + "/artm/"))
        print(subprocess.check_output('pwd', cwd=os.getcwd() + "/artm/wrapper"))
        print(subprocess.check_output('ls', cwd=os.getcwd() + "/artm/wrapper"))
        print("----")
        print(subprocess.check_output('pwd', cwd=dir_path + "/.."))
        print(subprocess.check_output('ls', cwd=dir_path + "/.."))
        print("-=-=-")
        print(subprocess.check_output('pwd', cwd=dir_path + "/../artm"))
        print(subprocess.check_output('ls', cwd=dir_path + "/../artm"))
        print("-=-=-")
        print(subprocess.check_output('pwd', cwd=dir_path + "/../artm/wrapper"))
        print(subprocess.check_output('ls', cwd=dir_path + "/../artm/wrapper"))

        if not os.path.exists(source_file):
            sys.stderr.write("Can't find required file: {}\n".format(
                source_file))
            sys.exit(-1)

        if not protoc_exec:
            raise ValueError("No protobuf compiler executable was found!")

        try:
            tmp_dir = tempfile.mkdtemp(dir=src_folder_abs)
            sys.stderr.write("tmp_dir {} exists: {}\n".format(tmp_dir, os.path.isdir(tmp_dir)))
            # tmp_dir = os.path.join(os.getcwd(), tmp_dir[2:])
            sys.stderr.write("tmp_dir {} exists: {}\n".format(tmp_dir, os.path.isdir(tmp_dir)))
            sys.stderr.write("proto_file {} exists: {}\n".format(source_file, os.path.isfile(source_file)))
            protoc_command = [
                protoc_exec,
                "-I=" + src_folder_abs,
                "--python_out=" + tmp_dir,
                source_file]
            print("Executing {}...".format(protoc_command))
            # protoc seems to not understand relative paths, so we need to tweak cwd
            # see https://github.com/protocolbuffers/protobuf/issues/3028
            if subprocess.call(protoc_command, cwd=src_folder_abs):
                raise
            src_py_file = src_proto_file.replace(".proto", "_pb2.py")
            if os.path.exists(dst_py_file_abs):
                os.remove(dst_py_file_abs)

            print("Moving {} to {}".format(os.path.join(tmp_dir, src_py_file), dst_py_file_abs))

            compiled_result = os.path.join(tmp_dir, src_py_file)
            sys.stderr.write("first file {} exists: {}\n".format(compiled_result, os.path.isfile(compiled_result)))

            # dst_dir = './artm/wrapper/'
            # sys.stderr.write("dst_dir {} exists: {}\n".format(dst_dir, os.path.isdir(dst_dir)))
            # print(subprocess.call('ls', cwd=dst_dir))

            os.rename(os.path.join(tmp_dir, src_py_file), dst_py_file_abs)
            raise ValueError()
        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)


if __name__ == "__main__":
        # Generate necessary .proto file if it doesn't exist.
        proto_name = "messages.proto"

        src_folder = "../src"
        dst_dir = './artm/wrapper/'

        src_folder = src_folder + "/artm"
        generate_proto_files(
            src_folder,
            proto_name,
            dst_dir + "messages_pb2.py")


