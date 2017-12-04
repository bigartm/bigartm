cmake_minimum_required (VERSION 2.8.0)

project(3rdparty)

if (MSVC)
  add_definitions("/w")
else (MSVC)
  add_definitions("-w")
endif (MSVC)

set(BUILD_TESTING "OFF" CACHE "" INTERNAL)

add_subdirectory(gflags EXCLUDE_FROM_ALL)

# Do not call find_package for gflags and fill all relevant info manually
# We cannot make glog link to gflags __target__, because cmake would try to
# pull it into glog's export set and will get confused and fail
# Instead we manually specify location and include path for gflags, this way
# cmake does not create inter-target dependency and export set does not fail
set(WITH_GFLAGS "OFF" CACHE "" INTERNAL)
set(HAVE_LIB_GFLAGS "ON")
set(gflags_NAMESPACE "gflags")
set(gflags_FOUND "ON")
get_target_property(gflags_INCLUDE_DIR gflags INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(gflags_LIBRARIES gflags INTERFACE_LINK_LIBRARIES)
set(gflags_LIBRARIES $<TARGET_FILE:gflags> ${gflags_LIBRARIES})

add_subdirectory(glog EXCLUDE_FROM_ALL)

# We need to ask CMake to compile gflags before compiling glog, so that
# ${gflags_LIBRARIES} actually exists
add_dependencies(glog gflags)

set(CMAKE_INSTALL_INCLUDEDIR include/artm)
add_subdirectory(protobuf-3.0.0/cmake EXCLUDE_FROM_ALL)
set(PROTOBUF_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf-3.0.0/cmake CACHE "" INTERNAL)
