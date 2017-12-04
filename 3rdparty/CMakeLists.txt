cmake_minimum_required (VERSION 2.8.0)

project(3rdparty)

if (MSVC)
  add_definitions("/w")
else (MSVC)
  add_definitions("-w")
endif (MSVC)

# This line enables install(EXPORT) in gflags
# If it's missing, glog complains about it later in the build process
# Nothing is actually installed due to ECLUDE_FROM_ALL
set(INSTALL_STATIC_LIBS "ON")
add_subdirectory(gflags EXCLUDE_FROM_ALL)

# We use target included directly from gflags, so we need to disable find_package for it
# And trick glog's CMakeLists into believing we have gflags
set(CMAKE_DISABLE_FIND_PACKAGE_gflags "ON")
set(gflags_FOUND "ON")
# This one tricks cmake into adding correct include dirs, dependent libraries, etc on target_link_libraries call in glog
set(gflags_LIBRARIES gflags)
add_subdirectory(glog EXCLUDE_FROM_ALL)

set(CMAKE_INSTALL_INCLUDEDIR include/artm)
add_subdirectory(protobuf-3.0.0/cmake EXCLUDE_FROM_ALL)
set(PROTOBUF_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf-3.0.0/cmake CACHE "" INTERNAL)
