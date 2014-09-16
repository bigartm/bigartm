#!/usr/bin/env python

from rpcz import compiler

compiler.generate_proto('../common/search.proto', '.')
compiler.generate_proto(
        '../common/search.proto', '.',
        with_plugin='python_rpcz', suffix='_rpcz.py',
        plugin_binary=
            '../../build/src/rpcz/plugin/python/protoc-gen-python_rpcz')
