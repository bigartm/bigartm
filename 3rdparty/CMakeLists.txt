cmake_minimum_required (VERSION 2.8.0)

project(3rdparty)

if (MSVC)
  add_definitions("/w")
else (MSVC)
  add_definitions("-w")
endif (MSVC)

# Disable tests and install rules for glog
# Enabled install rules will cause configuration failure
set(BUILD_TESTING "OFF" CACHE "" INTERNAL)
set(CMAKE_SKIP_INSTALL_RULES TRUE)

add_subdirectory(gflags EXCLUDE_FROM_ALL)

# Do not call find_package for gflags and pretend it exists
set(WITH_GFLAGS "OFF" CACHE "" INTERNAL)
set(HAVE_LIB_GFLAGS "ON")
set(gflags_NAMESPACE "gflags")
set(gflags_FOUND "ON")

# Make glog link to gflags target directly, cmake will handle the rest
set(gflags_LIBRARIES gflags)

add_subdirectory(glog EXCLUDE_FROM_ALL)

# We need to ask CMake to compile gflags before compiling glog, so that
# ${gflags_LIBRARIES} actually exists
add_dependencies(glog gflags)

set(CMAKE_SKIP_INSTALL_RULES FALSE)

set(CMAKE_INSTALL_INCLUDEDIR include/artm)
add_subdirectory(protobuf-3.0.0/cmake EXCLUDE_FROM_ALL)
set(PROTOBUF_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf-3.0.0/cmake CACHE "" INTERNAL)

# This is a hack. We build libprotobuf (a STATIC library) with __dllexport
# so that when we later link it into SHARED library artm.dll all protobuf
# symbols would get exported and cause no linking errors when using the
# library
target_compile_definitions(libprotobuf PRIVATE PROTOBUF_USE_DLLS LIBPROTOBUF_EXPORTS)
