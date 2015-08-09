"""
Implementation of wrapper API
"""

import os
import sys
import ctypes

from google import protobuf

from . import utils
from .exceptions import ARTM_EXCEPTION_BY_CODE
from .spec import ARTM_API


class LibArtm(object):
    def __init__(self, lib_name=None):
        self.cdll = self._load_cdll(lib_name)

        # adding specified functions
        for spec in ARTM_API:
            func = self.cdll[spec.name]
            setattr(self, spec.name, self._wrap_call(func, spec))
            # TODO: add docstring for wrapped function

    def _load_cdll(self, lib_name):
        # choose default library name
        default_lib_name = 'artm.so'
        if sys.platform.startswith('win'):
            default_lib_name = 'artm.dll'
        if sys.platform.startswith('darwin'):
            default_lib_name = 'artm.dylib'

        if lib_name is None:
            # try to get library path from environment variable
            lib_name = os.environ.get('ARTM_SHARED_LIBRARY')

        if lib_name is None:
            # set the default library name
            lib_name = default_lib_name

        try:
            cdll = ctypes.CDLL(lib_name)
        except OSError as e:
            exception_message = (
                e.message + '\n'
                'Failed to load artm shared library. '
                'Try to add the location of `{default_lib_name}` file into your PATH '
                'system variable, or to set ARTM_SHARED_LIBRARY - a specific system variable '
                'which may point to `{default_lib_name}` file, including the full path.'
            ).format(**locals())
            raise OSError(exception_message)

        return cdll

    def _check_error(self, error_code):
        if error_code < -1:
            self.cdll.ArtmGetLastErrorMessage.restype = ctypes.c_char_p
            error_message = self.cdll.ArtmGetLastErrorMessage()

            # remove exception name from error message
            error_message = error_message.split(':', 1)[-1].strip()

            exception_class = ARTM_EXCEPTION_BY_CODE.get(error_code)
            if exception_class is not None:
                raise exception_class(error_message)
            else:
                raise RuntimeError(error_message)

    def _copy_request_result(self, length):
        message_blob = ctypes.create_string_buffer(length)
        error_code = self.cdll.ArtmCopyRequestResult(length, message_blob)
        self._check_error(error_code)
        return message_blob

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

            cargs = []
            for (arg_index, arg), (arg_name, arg_type) in zip(enumerate(args), spec.arguments):
                # try to cast argument to the required type
                arg_casted = arg
                if issubclass(arg_type, protobuf.message.Message) and isinstance(arg, dict):
                    # dict -> protobuf message
                    arg_casted = utils.dict_to_message(arg, arg_type)

                # check argument type
                if not isinstance(arg_casted, arg_type):
                    raise TypeError('Argument {arg_index} ({arg_name}) should have '
                                    'type {arg_type} but {given_type} given'.format(
                        arg_index=arg_index,
                        arg_name=arg_name,
                        arg_type=str(arg_type),
                        given_type=str(type(arg)),
                    ))
                arg = arg_casted

                # construct c-style arguments                
                if issubclass(arg_type, basestring):
                    arg_cstr_p = ctypes.create_string_buffer(arg)
                    cargs.append(arg_cstr_p)

                elif issubclass(arg_type, protobuf.message.Message):
                    message_str = arg.SerializeToString()
                    message_cstr_p = ctypes.create_string_buffer(message_str)
                    cargs += [len(message_str), message_cstr_p]

                else:
                    cargs.append(arg)

            # make api call
            if spec.result_type is not None:
                func.restype = spec.result_type
            result = func(*cargs)
            self._check_error(result)

            # return result value
            if spec.request_type is not None:
                return self._copy_request_result(length=result)
            if spec.result_type is not None:
                return result

        return artm_api_call
