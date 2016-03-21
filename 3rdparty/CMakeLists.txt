cmake_minimum_required (VERSION 2.8.0)

project(3rdparty)

if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
   ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all")
endif()

add_subdirectory(gflags)
add_subdirectory(glog)

add_subdirectory(protobuf-cmake)

if(UNIX)
  add_subdirectory(protobuf)
endif(UNIX)
