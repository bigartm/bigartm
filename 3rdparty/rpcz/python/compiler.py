#!/usr/bin/env python

import os
import sys
from distutils import spawn


class CompilerException(Exception):
    pass


def generate_proto(source, output_dir,
                   with_plugin='python', suffix='_pb2.py', plugin_binary=None):
    """Invokes the Protocol Compiler to generate a _pb2.py from the given
    .proto file.  Does nothing if the output already exists and is newer than
    the input."""
    protoc = spawn.find_executable("protoc")
    if protoc is None:
        raise CompilerException(
            "protoc not found. Make sure that it is in the path.")

    output = os.path.join(
            output_dir,
            os.path.basename(source.replace(".proto", suffix)))

    if not os.path.exists(source):
        raise CompilerException("Can't find required file: " + source)

    if (os.path.exists(output) and
        os.path.getmtime(source) <= os.path.getmtime(output)):
        print "Generated proto %s is up-to-date." % output
        return

    print "Generating %s" % output

    protoc_command = protoc + ' -I "%s" --%s_out="%s" "%s"' % (
            os.path.dirname(source), with_plugin, output_dir, source)
    if plugin_binary:
        if os.path.exists(plugin_binary):
            protoc_command += ' --plugin=protoc-gen-%s=%s' % (with_plugin,
                                                              plugin_binary)
        else:
            print ("Plugin not found at '%s'. We are going to run protoc "
                   "anyway, and perhaps it will be able to find it in its "
                   "search path." % plugin_binary)
    if os.system(protoc_command) != 0:
        raise CompilerException(
            "Error occurred while running protoc.")
    else:
        print "Generated source successfully."
