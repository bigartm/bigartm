# Copyright 2017, Additive Regularization of Topic Models.

"""
Implementation of wrapper API
"""

import os
import sys
import ctypes

import numpy
import six
from six.moves import zip
from google import protobuf

from . import utils
from .exceptions import ARTM_EXCEPTION_BY_CODE
from .spec import ARTM_API


class LibArtm(object):
    def __init__(self, lib_name=None, logging_config=None):
        self.cdll, self.lib_name = self._load_cdll(lib_name)

        # adding specified functions
        for spec in ARTM_API:
            func = self.cdll[spec.name]
            setattr(self, spec.name, self._wrap_call(func, spec))
            # TODO: add docstring for wrapped function

        if logging_config is not None:
            self.ArtmConfigureLogging(logging_config)

    def __deepcopy__(self, memo):
        return self

    def _load_cdll(self, lib_name):
        # choose default library name
        default_lib_name = 'libartm.so'
        if sys.platform.startswith('win'):
            default_lib_name = 'artm.dll'
        if sys.platform.startswith('darwin'):
            default_lib_name = 'libartm.dylib'

        lib_names = []
        
        if lib_name is not None:
            lib_names.append(lib_name)
            
        env_lib_name = os.environ.get('ARTM_SHARED_LIBRARY')
        if env_lib_name is not None:
            lib_names.append(env_lib_name)
        
        lib_names.append(os.path.join(os.path.dirname(__file__), "..", default_lib_name))
        lib_names.append(default_lib_name)
        
        # We look into 4 places: lib_name, ARTM_SHARED_LIBRARY, packaged default_lib_name
        # and then default_lib_name
        cdll = None
        for ln in lib_names:
            try:
                cdll = ctypes.CDLL(ln)
                break
            except OSError as e:
                if ln == default_lib_name:
                    exc = e
                continue
        if cdll is None:
            exception_message = (
                '{exc}\n'
                'Failed to load artm shared library from `{lib_names}`. '
                'Try to add the location of `{default_lib_name}` file into your PATH '
                'system variable, or to set ARTM_SHARED_LIBRARY - the specific system variable '
                'which may point to `{default_lib_name}` file, including the full path.'
            ).format(**locals())
            raise OSError(exception_message)

        return (cdll, ln)

    def version(self):
        self.cdll.ArtmGetVersion.restype = ctypes.c_char_p
        ver = self.cdll.ArtmGetVersion()
        return ver if six.PY2 else ver.decode('utf-8')

    def _check_error(self, error_code):
        if error_code < -1:
            self.cdll.ArtmGetLastErrorMessage.restype = ctypes.c_char_p
            error_message = self.cdll.ArtmGetLastErrorMessage()
            if six.PY3:
                error_message = error_message.decode('utf-8')

            # remove exception name from error message
            error_message = error_message.split(':', 1)[-1].strip()

            exception_class = ARTM_EXCEPTION_BY_CODE.get(error_code)
            if exception_class is not None:
                raise exception_class(error_message)
            else:
                raise RuntimeError(error_message)

    def _get_requested_message(self, length, func):
        message_blob = ctypes.create_string_buffer(length)
        error_code = self.cdll.ArtmCopyRequestedMessage(length, message_blob)
        self._check_error(error_code)
        message = func()
        message.ParseFromString(message_blob.raw)
        return message

    def _wrap_call(self, func, spec):

        def artm_api_call(*args):
            # check the number of arguments
            n_args_given = len(args)
            n_args_takes = len(spec.arguments)
            if n_args_given != n_args_takes:
                raise TypeError('{func_name} takes {n_takes} argument ({n_given} given)'.format(
                    func_name=spec.name,
                    n_takes=n_args_takes,
                    n_given=n_args_given,
                ))

            c_args = []
            for (arg_pos, arg_value), (arg_name, arg_type) in zip(enumerate(args), spec.arguments):
                # try to cast argument to the required type
                arg_casted = arg_value
                if issubclass(arg_type, protobuf.message.Message) and isinstance(arg_value, dict):
                    # dict -> protobuf message
                    arg_casted = utils.dict_to_message(arg_value, arg_type)

                # check argument type
                if not isinstance(arg_casted, arg_type):
                    raise TypeError('Argument {arg_position} ({arg_name}) should have '
                                    'type {arg_type} but {given_type} given'.format(
                        arg_position=arg_pos,
                        arg_name=arg_name,
                        arg_type=str(arg_type),
                        given_type=str(type(arg_value)),
                    ))
                arg_value = arg_casted

                # construct c-style arguments
                if issubclass(arg_type, str):
                    arg_cstr_p = ctypes.create_string_buffer(arg_value.encode('utf-8'))
                    c_args.append(arg_cstr_p)

                elif issubclass(arg_type, protobuf.message.Message):
                    message_str = arg_value.SerializeToString()
                    message_cstr_p = ctypes.create_string_buffer(message_str)
                    c_args += [len(message_str), message_cstr_p]

                elif issubclass(arg_type, numpy.ndarray):
                    c_args += [ctypes.c_int64(arg_value.nbytes), ctypes.c_char_p(arg_value.ctypes.data)]

                else:
                    c_args.append(arg_value)

            # make api call
            if spec.result_type is not None:
                func.restype = spec.result_type
            result = func(*c_args)
            self._check_error(result)

            # return result value
            if spec.request_type is not None:
                return self._get_requested_message(length=result, func=spec.request_type)
            if spec.result_type is not None:
                return result

        return artm_api_call
