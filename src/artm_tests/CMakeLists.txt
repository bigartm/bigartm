project(artm_tests)

set(SRC_LIST
	api.cc
	batch_manager_test.cc
	blas_test.cc
	boost_thread_test.cc
	cache_manager_test.cc
	collection_parser_test.cc
	cpp_interface_test.cc
	master_model_test.cc
	multiple_classes_test.cc
	regularizers_test.cc
	scores_test.cc
	repeatable_result_test.cc
	supcry_test.cc
	template_manager_test.cc
	test_mother.cc
	thread_safe_holder_test.cc
	topic_seg_test.cc
	transactions_test.cc
	${3RD_PARTY_DIR}/gtest/fused-src/gtest/gtest_main.cc
	${3RD_PARTY_DIR}/gtest/fused-src/gtest/gtest-all.cc
)

add_definitions("-Dartm_EXPORTS")
add_executable(artm_tests ${SRC_LIST})
add_dependencies(artm_tests messages_proto)
target_link_libraries(artm_tests artm-static ${BOOST_IMPORTED_TARGETS})
target_include_directories(artm_tests PRIVATE
    ${3RD_PARTY_DIR}/gtest/fused-src/
    ${CMAKE_CURRENT_LIST_DIR}/../
    )

# add target for testing
add_test(NAME core_tests
    COMMAND artm_tests --gtest_output=xml:${CMAKE_CURRENT_BINARY_DIR}/junit.xml
    )
