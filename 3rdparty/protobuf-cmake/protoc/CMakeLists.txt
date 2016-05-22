project(protoc)

include(../protobuf-common.cmake)

set(SOURCES ${PROTOBUF_ROOT}/src/google/protobuf/compiler/main.cc)

add_definitions(-D_CONSOLE)

add_executable(protoc ${SOURCES})
target_link_libraries(protoc libprotoc libprotobuf)
if (UNIX)
  target_link_libraries(protoc -lpthread)
endif (UNIX)

set(PROTOBUF_PROTOC_EXECUTABLE protoc CACHE INTERNAL "protoc build")

set(PROTOBUF_PROTOC_EXECUTABLE_PATH
    ${CMAKE_CURRENT_BINARY_DIR}/protoc${CMAKE_EXECUTABLE_SUFFIX}
  CACHE
  FILEPATH
  "path to protoc executable")

if (MSVC)
  install(TARGETS protoc RUNTIME DESTINATION ${BIN_DIR})
elseif (UNIX)
  add_custom_command(TARGET protoc
    POST_BUILD
    COMMAND cp ${PROTOBUF_PROTOC_EXECUTABLE_PATH} ${CMAKE_SOURCE_DIR}/3rdparty/protobuf/src/)
endif (MSVC)
