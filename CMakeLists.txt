cmake_minimum_required(VERSION 3.3)

# CMake Policy 0060 (CMP0060) controls how CMake
# adds library dependencies to linker
# When this policy is disabled (turned to OLD),
# CMake uses weird linker options (e.g. -Wl,-Bstatic -lfoo -Wl,-dynamic),
# which can mess up linkage types of libraries
# When this policy is enabled (turned to NEW),
# CMake uses full path when linking libraries,
# and this helps to avoid aforementioned linker options
# The complete description of this policy can be found here:
# https://cmake.org/cmake/help/latest/policy/CMP0060.html
IF (POLICY CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif (POLICY CMP0060)

project(BigARTM)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(3RD_PARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty)

if (WIN32)
  add_definitions("-D_WIN32")
  add_definitions("-DWIN32")
endif (WIN32)

if (MSVC)
  add_definitions("-D_VARIADIC_MAX=10")
  add_definitions("-D_CRT_SECURE_NO_WARNINGS")
  add_definitions("-D_SCL_SECURE_NO_WARNINGS")
  add_definitions("/wd4251")
  add_definitions("/MP")
  add_definitions("/EHsc")
else (MSVC)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
  endif (NOT CMAKE_BUILD_TYPE)
  message("-- Build type: ${CMAKE_BUILD_TYPE}")
endif (MSVC)

option(BUILD_TESTS "Indicates whether to build artm_tests" ON)
option(BUILD_BIGARTM_CLI "Indicates whether to build bigartm-CLI executable" ON)
option(BUILD_INTERNAL_PYTHON_API "Indicates whether to build Python API" ON)

set(PYTHON python CACHE INTERNAL "Python command")

# Describe necessary options for static version of `bigartm` executable

include(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(
  BUILD_BIGARTM_CLI_STATIC "[EXPERIMENTAL] Requests static version of executable `bigartm`" OFF
  "BUILD_BIGARTM_CLI;UNIX;NOT APPLE" OFF)

CMAKE_DEPENDENT_OPTION(
  HAS_STATIC_LIBUNWIND "[EXPERIMENTAL] Indicates presence if static version of libunwind" OFF
  "BUILD_BIGARTM_CLI_STATIC" OFF)

if (BUILD_BIGARTM_CLI_STATIC AND NOT HAS_STATIC_LIBUNWIND)
  message(WARNING
    "You're requesting static version of CLI `bigartm`. "
    "Some 3rdparty of library (namely, `glog`) uses `libunwind` library, which "
    "provides convenient way of maniulating with call frames. "
    "Here `glog` uses this library mostly for stacktrace generation. "
    "Unfortunatley, not all Linux distributives provide static version of it "
    "in official repositories, that's why we disable usage of this library by default.\n"

    "If you certainly have static version of `libunwind` on your system "
    "and want to use it in BigARTM project, just set option "
    "'HAS_STATIC_LIBUNWIND' to 'ON'. For example, if you use CMake "
    "as a command-line utility, you can add option '-DHAS_STATIC_LIBUNWIND=ON' "
    "to argument list.")

  # There is problem with some Linux distributions (e.g. Fedora),
  # which currently doesn't provide convenient way to obtain
  # static version of `libunwind` library (either via official or custom repos)
  # This library is autodetected by 3rdparty `glog` module.
  # That's why there may be some problems when user wants to build
  # static CLI `bigartm` on system with only shared version of `libunwind`.
  #
  # It sounds reasonable to disable autodetection of `libunwind` library
  # by `glog` while building static CLI executable.
  # One possible way to do this is to "forge" `glog` module that
  # there is no libunwind library (via CMake CACHE vairables).
  # A nice candidate to do is variable "UNWIND_LIBRARY" which is directly
  # related to autodetection in file 3rdparty/glog/CMakeLists.txt
  set(UNWIND_LIBRARY CACHE INTERNAL "")
endif (BUILD_BIGARTM_CLI_STATIC AND NOT HAS_STATIC_LIBUNWIND)

if (APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif (APPLE)

set(BUILD_STATIC_LIBS ON)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_gflags_LIB ON)
set(BUILD_gflags_nothreads_LIB OFF)
set(protobuf_BUILD_TESTS OFF)

include(CheckIncludeFiles)
check_include_files(unistd.h HAVE_UNISTD_H)

# set compiler flags
if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
    ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -fPIC")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -Wall")
  set(GFLAGS_INTTYPES_FORMAT "C99")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_LIBS ON)
  set(GFLAGS_INTTYPES_FORMAT "VC7")
  include(InstallRequiredSystemLibraries)
else (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
      ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
  message(WARNING "BigARTM has not been tested with '${CMAKE_CXX_COMPILER_ID}' compiler.")
endif (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
       ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))

set(Boost_ADDITIONAL_VERSIONS "1.57" "1.57.0" "1.56" "1.56.0" "1.55" "1.55.0" "1.54" "1.54.0" "1.53" "1.53.0" "1.52" "1.52.0" "1.51" "1.51.0" "1.50" "1.50.0" "1.49" "1.49.0" "1.48" "1.48.0" "1.47" "1.47.0" "1.46" "1.46.0" "1.45" "1.45.0" "1.44" "1.44.0" "1.42" "1.42.0" "1.41.0" "1.41" "1.40.0" "1.40")

# find boost
find_package(Boost REQUIRED)
if (NOT Boost_FOUND)
  message(SEND_ERROR "Failed to find boost libraries.")
endif (NOT Boost_FOUND)

set(BIGARTM_BOOST_COMPONENTS thread program_options date_time filesystem iostreams system)
if (Boost_VERSION GREATER 104700)
  set(BIGARTM_BOOST_COMPONENTS ${BIGARTM_BOOST_COMPONENTS} chrono timer)
endif (Boost_VERSION GREATER 104700)

find_package(Boost COMPONENTS REQUIRED ${BIGARTM_BOOST_COMPONENTS})
if (NOT Boost_FOUND)
  message(SEND_ERROR "Failed to find required boost libraries.")
  return()
endif (NOT Boost_FOUND)

set(BOOST_IMPORTED_TARGETS Boost::boost)
foreach(boost_component ${BIGARTM_BOOST_COMPONENTS})
    list(APPEND BOOST_IMPORTED_TARGETS Boost::${boost_component})
endforeach(boost_component)

add_subdirectory(3rdparty)  # gflags must be compiled without -std=c++11

if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
    ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  else (COMPILER_SUPPORTS_CXX11)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  endif (COMPILER_SUPPORTS_CXX11)
endif (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
       ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))

add_subdirectory(src)
add_subdirectory(python)

if (MSVC)
  install(FILES LICENSE.txt DESTINATION .)
endif (MSVC)
