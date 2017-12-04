cmake_minimum_required (VERSION 2.8.0)

project(3rdparty)

if (MSVC)
  add_definitions("/w")
else (MSVC)
  add_definitions("-w")
endif (MSVC)

# We do not want to build unit tests for gflags and glog
set(BUILD_TESTING_BACKUP ${BUILD_TESTING})
set(BUILD_TESTING "OFF")

add_subdirectory(gflags)

# We use target included directly from gflags, so we need to disable find_package for it
# And trick glog's CMakeLists into believing we have gflags
set(CMAKE_DISABLE_FIND_PACKAGE_gflags "ON")
set(gflags_FOUND "ON")
# This one tricks cmake into adding correct include dirs, dependent libraries, etc on target_link_libraries call in glog
set(gflags_LIBRARIES gflags)
add_subdirectory(glog)

set(BUILD_TESTING ${BUILD_TESTING_BACKUP})

add_subdirectory(protobuf-3.0.0/cmake)
