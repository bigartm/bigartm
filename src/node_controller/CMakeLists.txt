project(node_controller)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	node_controller.cc
	${CMAKE_SOURCE_DIR}/src/artm/cpp_interface.cc
	${CMAKE_SOURCE_DIR}/src/artm/messages.pb.cc
)


add_executable(node_controller ${SRC_LIST})
target_link_libraries(node_controller artm)

install(TARGETS node_controller DESTINATION bin)