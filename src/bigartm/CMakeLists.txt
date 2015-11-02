project(bigartm)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	srcmain.cc
	${CMAKE_SOURCE_DIR}/src/artm/cpp_interface.cc
	${CMAKE_SOURCE_DIR}/src/artm/messages.pb.cc
)

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC ON)
find_package(Boost COMPONENTS REQUIRED ${BIGARTM_BOOST_COMPONENTS})

add_executable(bigartm ${SRC_LIST})
set(CMAKE_EXE_LINKER_FLAGS "-static")
target_link_libraries(bigartm
	artm-static
	${PROTOBUF_LIBRARIES}
	${GLOG_LIBRARIES}
	${Boost_LIBRARIES})

install(TARGETS bigartm DESTINATION bin)
