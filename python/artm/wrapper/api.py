"""
Implementation of wrapper API
"""

import ctypes

from google import protobuf

from . import utils
from .exceptions import ARTM_EXCEPTION_BY_CODE
from .spec import ARTM_API


class LibArtm(object):
    def __init__(self, lib_name):
        self.cdll = ctypes.CDLL(lib_name)
        self._spec_by_name = {spec.name: spec for spec in ARTM_API}

    def __getattr__(self, name):
        spec = self._spec_by_name.get(name)
        if spec is None:
            raise AttributeError('%s is not a function of libartm' % name)
        func = getattr(self.cdll, name)
        return self._wrap_call(func, spec)

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
        error_code = self.lib_.ArtmCopyRequestResult(length, message_blob)
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
