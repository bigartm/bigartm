from cpython cimport Py_DECREF, Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free


cdef extern from "Python.h":
    void PyEval_InitThreads()


cdef extern from "rpcz/connection_manager.hpp" namespace "rpcz":
    cdef void install_signal_handler()


cdef extern from "rpcz/callback.hpp" namespace "rpcz":
  pass


def init():
    import sys
    # InstallSignalHandler()
    PyEval_InitThreads()


init()


cdef extern from "string" namespace "std":
    cdef cppclass string:
        string()
        string(char*)
        string(char*, size_t)
        size_t size()
        char* c_str()


cdef string make_string(pystring) except *:
    return string(pystring, len(pystring))


cdef string_ptr_to_pystring(string* s):
    return s.c_str()[:s.size()]


cdef cstring_to_pystring(void* s, size_t size):
    return (<char*>s)[:size]


cdef string_to_pystring(string s):
    return s.c_str()[:s.size()]


cdef extern from "rpcz/sync_event.hpp" namespace "rpcz":
    cdef cppclass _sync_event "rpcz::sync_event":
        void signal() nogil
        void wait() nogil

cdef extern from "rpcz/rpc.hpp" namespace "rpcz":
    cdef cppclass _rpc "rpcz::rpc":
        bint ok()
        int get_status()
        string get_error_message()
        int get_application_error_code()
        long get_deadline_ms()
        void set_deadline_ms(long)
        int wait() nogil


cdef class WrappedRPC:
    cdef _rpc *thisptr
    cdef _sync_event *sync_event

    def __cinit__(self):
        self.thisptr = new _rpc()
        self.sync_event = new _sync_event()
    def __dealloc__(self):
        del self.sync_event
        del self.thisptr
    def ok(self):
        return self.thisptr.ok()
    def wait(self):
        with nogil:
            self.sync_event.wait()

    property status:
        def __get__(self):
            return self.thisptr.get_status()
    property application_error_code:
        def __get__(self):
            return self.thisptr.get_application_error_code()
    property error_message:
        def __get__(self):
            return string_to_pystring(self.thisptr.get_error_message())
    property deadline_ms:
        def __get__(self):
            return self.thisptr.get_deadline_ms()
        def __set__(self, value):
            self.thisptr.set_deadline_ms(value)


cdef struct ClosureWrapper:
    string* response_str
    void* response_obj
    void* callback
    void* rpc


cdef extern from "rpcz/macros.hpp" namespace "rpcz":
    cdef cppclass closure:
        pass

    closure* new_callback(void(ClosureWrapper*) nogil, ClosureWrapper*)


# this function is called from C++ after we gave up the GIL. We use "with gil"
# to acquire it.
cdef void python_callback_bridge(ClosureWrapper *closure_wrapper) with gil:
    (<object>closure_wrapper.response_obj).ParseFromString(
            string_ptr_to_pystring(closure_wrapper.response_str))
    response = <object>closure_wrapper.response_obj;
    callback = <object>closure_wrapper.callback
    rpc = <WrappedRPC>closure_wrapper.rpc
    rpc.sync_event.signal()
    if callback is not None:
        callback(response, rpc)
    Py_DECREF(<object>closure_wrapper.response_obj)
    Py_DECREF(<object>closure_wrapper.callback)
    Py_DECREF(<object>closure_wrapper.rpc)
    del closure_wrapper.response_str
    free(closure_wrapper)


cdef extern from "rpcz/rpc_channel.hpp" namespace "rpcz":
    cdef cppclass _rpc_channel "rpcz::rpc_channel":
        void call_method0(string service_name, string method_name,
                          string request, string* response, _rpc* rpc, 
                          closure* callback) except +


cdef class RpcChannel:
    cdef _rpc_channel *thisptr
    def __dealloc__(self):
        del self.thisptr
    def __init__(self):
        raise TypeError("Use Application.create_rpc_channel to create a "
                        "RpcChannel.")
    def call_method(self, service_name, method_name,
                   request, response, WrappedRPC rpc, callback):
        cdef ClosureWrapper* closure_wrapper = <ClosureWrapper*>malloc(
                sizeof(ClosureWrapper))
        closure_wrapper.response_str = new string()
        closure_wrapper.response_obj = <void*>response
        closure_wrapper.callback = <void*>callback
        closure_wrapper.rpc = <void*>rpc
        Py_INCREF(response)
        Py_INCREF(callback)
        Py_INCREF(rpc)
        self.thisptr.call_method0(
                make_string(service_name),
                make_string(method_name),
                make_string(request.SerializeToString()),
                closure_wrapper.response_str,
                rpc.thisptr,
                new_callback(
                    python_callback_bridge, closure_wrapper))


cdef extern from "rpcz/service.hpp" namespace "rpcz":
  cdef cppclass _server_channel "rpcz::server_channel":
    void send_error(int, string)
    void send0(string)
 

cdef class ServerChannel:
    cdef _server_channel *thisptr
    def __init__(self):
        raise TypeError("Do not initialize directly.")
    def __dealloc__(self):
        del self.thisptr
    def send_error(self, application_error_code, error_string=""):
      self.thisptr.send_error(application_error_code,
                             make_string(error_string))
      del self.thisptr
      self.thisptr = NULL
    def send(self, message):
      self.thisptr.send0(make_string(message.SerializeToString()))
      del self.thisptr
      self.thisptr = NULL


ctypedef void(*Handler)(user_data, string method,
                        void* payload, size_t payload_len,
                        _server_channel* channel) nogil


cdef void rpc_handler_bridge(user_data, string& method,
                             void* payload, size_t payload_len,
                             _server_channel* channel) with gil:
  cdef ServerChannel channel_ = ServerChannel.__new__(ServerChannel)
  channel_.thisptr = channel
  user_data._call_method(string_to_pystring(method),
                         cstring_to_pystring(payload, payload_len),
                         channel_)


cdef extern from "python_rpc_service.hpp" namespace "rpcz":
    cdef cppclass PythonRpcService:
        PythonRpcService(Handler, object)


cdef extern from "rpcz/rpcz.hpp" namespace "rpcz":
    cdef cppclass _application_options "rpcz::application::options":
        _application_options()
        int connection_manager_threads

    cdef cppclass _application "rpcz::application":
        _application()
        _application(_application_options options)
        _rpc_channel* create_rpc_channel(string)
        void terminate()
        void run() nogil


cdef class Application:
    cdef _application *thisptr

    def __cinit__(self, connection_manager_threads=None):
        cdef _application_options opts
        if connection_manager_threads:
            opts.connection_manager_threads = connection_manager_threads
        self.thisptr = new _application(opts)
    def __dealloc__(self):
        del self.thisptr
    def create_rpc_channel(self, endpoint):
        cdef RpcChannel channel = RpcChannel.__new__(RpcChannel)
        channel.thisptr = self.thisptr.create_rpc_channel(make_string(endpoint))
        return channel
    def terminate(self):
        self.thisptr.terminate()
    def run(self):
        with nogil:
            self.thisptr.run()


cdef extern from "rpcz/rpcz.hpp" namespace "rpcz":
    cdef cppclass _server "rpcz::server":
        _server(_application&)
        void register_service(PythonRpcService*, string name)
        void bind(string endpoint)


cdef class Server:
    cdef _server *thisptr
    def __cinit__(self, Application application):
        self.thisptr = new _server(deref(application.thisptr))
    def __dealloc__(self):
        del self.thisptr
    def register_service(self, service, name=None):
        cdef PythonRpcService* rpc_service = new PythonRpcService(
            rpc_handler_bridge, service)
        self.thisptr.register_service(rpc_service, make_string(name))
    def bind(self, endpoint):
        self.thisptr.bind(make_string(endpoint))


