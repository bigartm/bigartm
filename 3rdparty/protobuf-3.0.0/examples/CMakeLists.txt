# Minimum CMake required
cmake_minimum_required(VERSION 2.8.12)

# Project
project(protobuf-examples)

# Find required protobuf package
find_package(protobuf CONFIG REQUIRED)

if(protobuf_VERBOSE)
  message(STATUS "Using Protocol Buffers ${Protobuf_VERSION}")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

foreach(example add_person list_people)
  set(${example}_SRCS ${example}.cc)
  set(${example}_PROTOS addressbook.proto)

  #Code Generation
  if(protobuf_MODULE_COMPATIBLE) #Legacy Support
    protobuf_generate_cpp(${example}_PROTO_SRCS ${example}_PROTO_HDRS ${${example}_PROTOS})
    list(APPEND ${example}_SRCS ${${example}_PROTO_SRCS} ${${example}_PROTO_HDRS})
  else()

    foreach(proto_file ${${example}_PROTOS})
      get_filename_component(proto_file_abs ${proto_file} ABSOLUTE)
      get_filename_component(basename ${proto_file} NAME_WE)
      set(generated_files ${basename}.pb.cc ${basename}.pb.h)
      list(APPEND ${example}_SRCS ${generated_files})

      add_custom_command(
        OUTPUT ${generated_files}
        COMMAND protobuf::protoc
        ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR} ${proto_file_abs}
        COMMENT "Generating ${generated_files} from ${proto_file}"
        VERBATIM
      )
    endforeach()
  endif()

  #Executable setup
  set(executable_name ${example}_cpp)
  add_executable(${executable_name} ${${example}_SRCS} ${${example}_PROTOS})
  if(protobuf_MODULE_COMPATIBLE) #Legacy mode
    target_include_directories(${executable_name} PUBLIC ${PROTOBUF_INCLUDE_DIRS})
    target_link_libraries(${executable_name} ${PROTOBUF_LIBRARIES})
  else()
    target_link_libraries(${executable_name} protobuf::libprotobuf)
  endif()

endforeach()
