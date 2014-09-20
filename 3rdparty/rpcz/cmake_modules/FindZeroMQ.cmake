# - Try to find libzmq
# Once done, this will define
#
#  ZeroMQ_FOUND - system has libzmq
#  ZeroMQ_INCLUDE_DIRS - the libzmq include directories
#  ZeroMQ_LIBRARIES - link these to use libzmq

include(LibFindMacros)

set(ZEROMQ_INCLUDE_DIR ${3RD_PARTY_DIR}/zeromq/include ${3RD_PARTY_DIR}/cppzmq)
set(ZEROMQ_INCLUDE_DIRS ${ZEROMQ_INCLUDE_DIR})

set(ZEROMQ_LIBRARIES ${ZEROMQ_LIBRARY})

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(ZeroMQ_PROCESS_INCLUDES ZEROMQ_INCLUDE_DIR ZEROMQ_INCLUDE_DIRS)
set(ZeroMQ_PROCESS_LIBS ZEROMQ_LIBRARY ZEROMQ_LIBRARIES)
libfind_process(ZeroMQ)
