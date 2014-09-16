include(rpcz_functions)
find_package(ProtobufPlugin REQUIRED)

PROTOBUF_GENERATE_PYTHON(SEARCH_PB_PY_SRCS search.proto)
PROTOBUF_GENERATE_PYTHON_RPCZ(SEARCH_PB_PYRPCZ_SRCS search.proto)

PROTOBUF_GENERATE_CPP(SEARCH_PB_SRCS SEARCH_PB_HDRS search.proto)
PROTOBUF_GENERATE_RPCZ(SEARCH_RPCZ_SRCS SEARCH_RPCZ_HDRS search.proto)

add_library(search_pb ${SEARCH_PB_SRCS} ${SEARCH_PB_HDRS} ${SEARCH_RPCZ_SRCS}
                      ${SEARCH_RPCZ_HDRS})
target_link_libraries(search_pb ${PROTOBUF_LIBRARY})

add_custom_target(_force_python_protos ALL DEPENDS ${SEARCH_PB_PY_SRCS}
    ${SEARCH_PB_PYRPCZ_SRCS})
