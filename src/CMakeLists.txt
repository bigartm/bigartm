project(artm)

# find Boost

set(Boost_ADDITIONAL_VERSIONS "1.56" "1.56.0" "1.55" "1.55.0" "1.54" "1.54.0" "1.53" "1.53.0" "1.52" "1.52.0" "1.51" "1.51.0" "1.50" "1.50.0" "1.49" "1.49.0" "1.48" "1.48.0" "1.47" "1.47.0" "1.46" "1.46.0" "1.45" "1.45.0" "1.44" "1.44.0" "1.42" "1.42.0" "1.41.0" "1.41" "1.40.0" "1.40")

find_package(Boost REQUIRED)
if (NOT Boost_FOUND)
  message(SEND_ERROR "Failed to find boost libraries.")
endif()

set(BIGARTM_BOOST_COMPONENTS thread program_options date_time filesystem iostreams system)
if(Boost_VERSION GREATER 104700)
  set(BIGARTM_BOOST_COMPONENTS ${BIGARTM_BOOST_COMPONENTS} chrono timer)
endif()

find_package(Boost COMPONENTS REQUIRED ${BIGARTM_BOOST_COMPONENTS})
if(NOT Boost_FOUND)
  message(SEND_ERROR "Failed to find required boost libraries.")
  return()
else()
  include_directories(${Boost_INCLUDE_DIRS})
endif()

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