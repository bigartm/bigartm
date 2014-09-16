cmake_minimum_required(VERSION 2.8)

project(BigARTM)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

if (WIN32)
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

if (WIN32)
  add_definitions("-D_WIN32")
  add_definitions("-DWIN32")
endif()

if (MSVC)
  add_definitions("-D_VARIADIC_MAX=10")
  add_definitions("/wd4251")
  add_definitions("/MP")
  add_definitions("-D_CRT_SECURE_NO_WARNINGS")
  add_definitions("-D_SCL_SECURE_NO_WARNINGS")
else()
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
  endif()
  message("-- Build type: ${CMAKE_BUILD_TYPE}")
endif()

# set compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -fPIC -std=c++11")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(BUILD_STATIC_LIBS ON)
set(BUILD_SHARED_LIBS OFF)

if (MSVC)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_LIBS ON)
endif()

set(BUILD_gflags_LIB ON)
set(BUILD_gflags_nothreads_LIB OFF)
set(rpcz_build_static ON)
set(rpcz_build_shared ON)

set(3RD_PARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty)

add_subdirectory(3rdparty)
add_subdirectory(src)

if(MSVC)
  include(InstallRequiredSystemLibraries)
endif()

install(FILES LICENSE DESTINATION .)