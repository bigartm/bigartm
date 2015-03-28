project(artm_tests)

include_directories(${3RD_PARTY_DIR}/gtest/fused-src/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	batch_manager_test.cc
	blas_test.cc
	boost_thread_test.cc
	collection_parser_test.cc
	cpp_interface_test.cc
	instance_test.cc
	multiple_classes_test.cc
	nodes_connectivity_test.cc
	repeatable_result_test.cc
	rpcz_canary_service.pb.cc
	rpcz_canary_service.rpcz.cc
	rpcz_canary_test.cc
	template_manager_test.cc
	test_mother.cc
	thread_safe_holder_test.cc
	topic_model_test.cc
	${3RD_PARTY_DIR}/gtest/fused-src/gtest/gtest_main.cc
	${3RD_PARTY_DIR}/gtest/fused-src/gtest/gtest-all.cc
	${CMAKE_SOURCE_DIR}/src/artm/c_interface.cc
	${CMAKE_SOURCE_DIR}/src/artm/cpp_interface.cc
	${CMAKE_SOURCE_DIR}/src/artm/messages.pb.cc
)

add_definitions("-Dartm_EXPORTS")
add_executable(artm_tests ${SRC_LIST})
target_link_libraries(artm_tests artm-static
                                 rpcz_static
                                 ${LINK_DL}
                                 ${Boost_LIBRARIES}
                                 ${PROTOBUF_LIBRARIES}
                                 ${GLOG_LIBRARIES}
                                 ${ZeroMQ_LIBRARIES})
