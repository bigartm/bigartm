project(artm)

# find Protobuf
find_package(ProtobufPlugin REQUIRED)
if (NOT PROTOBUF_FOUND)
  message(SEND_ERROR "Failed to find protobuf.")
  return()
endif (NOT PROTOBUF_FOUND)

add_subdirectory(artm)

IF (BUILD_TESTS)
  add_subdirectory(artm_tests)
ENDIF (BUILD_TESTS)

if (BUILD_BIGARTM_CLI)
  add_subdirectory(bigartm)
endif (BUILD_BIGARTM_CLI)
