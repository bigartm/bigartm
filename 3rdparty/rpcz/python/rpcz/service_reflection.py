#!/usr/bin/env python
import rpcz.rpc
from rpcz import rpcz_pb2


def _call_method(service, method, request_proto, channel):
    method_descriptor = service._exposed_methods.get(method)
    if method_descriptor is None:
        channel.send_error(rpcz_pb2.rpc_response_header.NO_SUCH_METHOD)
        return
    handler = getattr(service, method, service._default_handler)
    request = method_descriptor.input_type._concrete_class()
    # TODO: error handling
    request.ParseFromString(request_proto)
    handler(request, channel)     
         

def _default_handler(service, request, channel):
    channel.send_error(rpcz_pb2.rpc_response_header.METHOD_NOT_IMPLEMENTED)

    
class GeneratedServiceType(type):
    def __new__(cls, name, bases, attrs):
        if 'DESCRIPTOR' not in attrs:
            # In subclasses, DESCRIPTOR is unavailable.
            return super(GeneratedServiceType, cls).__new__(cls, name, bases,
                                                            attrs)
        attrs['_call_method'] = _call_method
        descriptor = attrs['_descriptor'] = attrs['DESCRIPTOR']
        methods = attrs['_exposed_methods'] = {}
        for method in descriptor.methods:
            methods[method.name] = method
        attrs['_default_handler'] = _default_handler
        return super(GeneratedServiceType, cls).__new__(cls, name, bases,
                                                        attrs)


def _BuildStubMethod(method_descriptor):
    def call(stub, request, rpc=None, callback=None,
             deadline_ms=None):
        response = method_descriptor.output_type._concrete_class()
        if rpc is None:
            blocking_mode = True
            rpc = rpcz.rpc.RPC(deadline_ms = deadline_ms)
        else:
            blocking_mode = False
            if deadline_ms is not None:
              raise ValueError("'rpc' and 'deadline_ms' cannot be both "
                               "specified. Use rpc.deadline_ms to set a "
                               "deadline")
        stub._channel.call_method(stub._service_name,
                                  method_descriptor.name,
                                  request, response, rpc, callback)
        if blocking_mode:
            rpc.wait()
            return response
    return call


def _StubInitMethod(stub, channel, service_name=None):
    stub._channel = channel
    stub._service_name = service_name or stub.DESCRIPTOR.name


class GeneratedServiceStubType(GeneratedServiceType):
    def __new__(cls, name, bases, attrs):
        descriptor = attrs['DESCRIPTOR']
        attrs['__init__'] = _StubInitMethod
        for method in descriptor.methods:
            attrs[method.name] = _BuildStubMethod(method)
        return super(GeneratedServiceStubType, cls).__new__(cls, name, bases,
                                                            attrs)
