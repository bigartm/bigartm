cmake_minimum_required (VERSION 2.8.0)

project(3rdparty)

if (MSVC)
  add_definitions("/w")
else (MSVC)
  add_definitions("-w")
endif (MSVC)

add_subdirectory(gflags)
add_subdirectory(glog)

add_subdirectory(protobuf-cmake)
