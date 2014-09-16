project(RPCZ_GENERIC)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake_modules)

option(rpcz_build_tests "Build rpcz's tests." OFF)
option(rpcz_build_static "Build static library." OFF)
option(rpcz_build_examples "Build rpcz's examples." OFF)

find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread program_options date_time system chrono)
find_package(ZeroMQ REQUIRED)

include_directories(${ZEROMQ_INCLUDE_DIRS})
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/src)

cmake_minimum_required (VERSION 2.8.0)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -Werror")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DZMQ_STATIC")
endif(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_OSX_ARCHITECTURES x86_64)

set(rpcz_build_static ON)

add_subdirectory(src)

if(rpcz_build_tests)
  enable_testing()
  add_subdirectory(test)
endif(rpcz_build_tests)

if(rpcz_build_examples)
  add_subdirectory(examples/cpp)
endif(rpcz_build_examples)
