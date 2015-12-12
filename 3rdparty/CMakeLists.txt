cmake_minimum_required (VERSION 2.8.0)

project(3rdparty)

add_subdirectory(gflags)
add_subdirectory(glog)

add_subdirectory(protobuf-cmake)

if(UNIX)
  add_subdirectory(protobuf)
endif(UNIX)
