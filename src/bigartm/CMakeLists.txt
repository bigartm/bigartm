project(bigartm)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	srcmain.cc
	cooccurrence_dictionary.cc
	${CMAKE_SOURCE_DIR}/src/artm/cpp_interface.cc
)

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC ON)
find_package(Boost COMPONENTS REQUIRED ${BIGARTM_BOOST_COMPONENTS})

add_executable(bigartm ${SRC_LIST})
add_dependencies(bigartm messages_proto)

if (BUILD_BIGARTM_CLI_STATIC)
  set(CMAKE_EXE_LINKER_FLAGS "-static")
endif (BUILD_BIGARTM_CLI_STATIC)

if (WIN32)
  target_link_libraries(bigartm artm ${Boost_LIBRARIES})
else (WIN32)
  target_link_libraries(bigartm
    messages_proto
    artm-static
    ${PROTOBUF_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${Boost_LIBRARIES})
endif (WIN32)

install(TARGETS bigartm DESTINATION bin)
