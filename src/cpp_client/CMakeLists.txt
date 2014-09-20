project(cpp_client)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	srcmain.cc
	${CMAKE_SOURCE_DIR}/src/artm/cpp_interface.cc
	${CMAKE_SOURCE_DIR}/src/artm/messages.pb.cc
)

add_executable(cpp_client ${SRC_LIST})
target_link_libraries(cpp_client artm ${Boost_TIMER_LIBRARIES})
