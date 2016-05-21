project(artm)

# find Protobuf
find_package(ProtobufPlugin REQUIRED)
if (NOT PROTOBUF_FOUND)
  message(SEND_ERROR "Failed to find protobuf.")
  return()
else (NOT PROTOBUF_FOUND)
  include_directories(${PROTOBUF_INCLUDE_DIRS})
endif (NOT PROTOBUF_FOUND)

# find Glog
find_package(Glog REQUIRED)
if (NOT GLOG_FOUND)
  message(SEND_ERROR "Failed to find glog.")
  return()
else (NOT GLOG_FOUND)
  include_directories(${GLOG_INCLUDE_DIRS})
endif (NOT GLOG_FOUND)

# can it be avoided?
include_directories(${GFLAGS_INCLUDE_DIRS})

add_subdirectory(artm)
add_subdirectory(artm_tests)
add_subdirectory(bigartm)
