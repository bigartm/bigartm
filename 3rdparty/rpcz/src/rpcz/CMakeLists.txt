project (RPCZ)

# The version number.
set (RPCZ_VERSION_MAJOR 0)
set (RPCZ_VERSION_MINOR 9)

add_subdirectory(plugin)

protobuf_generate_cpp(RPCZ_PB_SRCS RPCZ_PB_HDRS proto/rpcz.proto)
set(PROTO_SOURCES ${RPCZ_PB_SRCS} ${RPCZ_PB_HDRS})

include_directories(${PROJECT_BINARY_DIR})

set(RPCZ_SOURCES
    application.cc clock.cc connection_manager.cc
    reactor.cc rpc.cc rpc_channel_impl.cc server.cc
    sync_event.cc zmq_utils.cc
    ${PROTO_SOURCES})
set(RPCZ_LIB_DEPS ${ZeroMQ_LIBRARIES}
                  ${PROTOBUF_LIBRARY}
                  ${Boost_THREAD_LIBRARIES}
		  ${Boost_DATE_TIME_LIBRARIES}
		  ${Boost_SYSTEM_LIBRARIES}
		  ${Boost_CHRONO_LIBRARIES}
		  )

if (rpcz_build_shared)
    add_library(rpcz SHARED ${RPCZ_SOURCES})
    target_link_libraries(rpcz ${RPCZ_LIB_DEPS})
    set(RPCZ_TARGET_LIBS rpcz)
endif(rpcz_build_shared)

if(rpcz_build_static)
    add_library(rpcz_static STATIC ${RPCZ_SOURCES})
    target_link_libraries(rpcz_static ${RPCZ_LIB_DEPS})
    set_target_properties(rpcz_static PROPERTIES OUTPUT_NAME rpcz)
    list(APPEND RPCZ_TARGET_LIBS rpcz_static)
endif(rpcz_build_static)

if (rpcz_build_zsendrp)
    add_executable(zsendrpc zsendrpc.cc)
    target_link_libraries(zsendrpc rpcz ${Boost_PROGRAM_OPTIONS_LIBRARIES})
endif(rpcz_build_zsendrp)
