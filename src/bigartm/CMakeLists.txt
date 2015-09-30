project(bigartm)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	srcmain.cc
	${CMAKE_SOURCE_DIR}/src/artm/cpp_interface.cc
	${CMAKE_SOURCE_DIR}/src/artm/messages.pb.cc
)

add_executable(bigartm ${SRC_LIST})
target_link_libraries(bigartm artm ${Boost_LIBRARIES})

install(TARGETS bigartm DESTINATION bin)
