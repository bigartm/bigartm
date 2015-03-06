project(artm)

# find Protobuf
find_package(ProtobufPlugin REQUIRED)
if(NOT PROTOBUF_FOUND)
  message(SEND_ERROR "Failed to find protobuf.")
  return()
else()
  include_directories(${PROTOBUF_INCLUDE_DIRS})
endif()

# find Glog
find_package(Glog REQUIRED)
if(NOT GLOG_FOUND)
  message(SEND_ERROR "Failed to find glog.")
  return()
else()
  include_directories(${GLOG_INCLUDE_DIRS})
endif()

# find ZEROMQ
find_package(ZeroMQ REQUIRED)
if(NOT ZeroMQ_FOUND)
  message(SEND_ERROR "Failed to find zeromq.")
  return()
else()
  include_directories(${ZEROMQ_INCLUDE_DIRS})
endif()

# can it be avoided?
include_directories(${3RD_PARTY_DIR}/rpcz/include)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/rpcz/src)
include_directories(${GFLAGS_INCLUDE_DIRS})

add_subdirectory(artm)
add_subdirectory(artm_tests)
add_subdirectory(cpp_client)
add_subdirectory(node_controller)

add_dependencies(artm-static rpcz)

install(FILES python/artm/library.py python/artm/messages_pb2.py python/artm/__init__.py DESTINATION python/artm)
install(FILES ../datasets/docword.kos.txt ../datasets/vocab.kos.txt DESTINATION python/examples)

FILE(GLOB PYTHON_EXAMPLES python/examples/*.py)
install(FILES ${PYTHON_EXAMPLES} DESTINATION python/examples)

FILE(GLOB PYTHON_EXPERIMENTS python/experiments/*.py)
install(FILES ${PYTHON_EXPERIMENTS} DESTINATION python/experiments)
