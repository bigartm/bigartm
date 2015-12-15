include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

# Static 'artm' library

set(SRC_LIST
	c_interface.cc
	c_interface.h
	messages.pb.cc
	messages.pb.h
	regularizer_interface.cc
	regularizer_interface.h
	score_calculator_interface.cc
	score_calculator_interface.h
	core/batch_manager.cc
	core/batch_manager.h
	core/cache_manager.cc
	core/cache_manager.h
	core/call_on_destruction.h
	core/collection_parser.cc
	core/collection_parser.h
	core/common.h
	core/data_loader.cc
	core/data_loader.h
	core/dense_phi_matrix.cc
	core/dense_phi_matrix.h
	core/dictionary.cc
	core/dictionary.h
	core/exceptions.h
	core/helpers.cc
	core/helpers.h
	core/instance.cc
	core/instance.h
	core/instance_schema.cc
	core/instance_schema.h
	core/internals.pb.cc
	core/internals.pb.h
	core/master_component.cc
	core/master_component.h
	core/merger.cc
	core/merger.h
	core/processor.cc
	core/processor.h
	core/processor_input.cc
	core/processor_input.h
	core/protobuf_helpers.h
	core/phi_matrix.h
	core/phi_matrix_operations.cc
	core/phi_matrix_operations.h
	core/scores_merger.cc
	core/scores_merger.h
	core/sync_event.cc
	core/sync_event.h
	core/template_manager.h
	core/thread_safe_holder.h
	core/topic_model.cc
	core/topic_model.h
	regularizer/decorrelator_phi.cc
	regularizer/decorrelator_phi.h
	regularizer/multilanguage_phi.cc
	regularizer/multilanguage_phi.h
	regularizer/smooth_sparse_phi.cc
	regularizer/smooth_sparse_phi.h
	regularizer/label_regularization_phi.cc
	regularizer/label_regularization_phi.h
	regularizer/smooth_sparse_theta.cc
	regularizer/smooth_sparse_theta.h
	regularizer/smooth_ptdw.cc
	regularizer/smooth_ptdw.h
	regularizer/specified_sparse_phi.cc
	regularizer/specified_sparse_phi.h
	regularizer/improve_coherence_phi.cc
	regularizer/improve_coherence_phi.h
	core/transform_function.h
	core/transform_function.cc
	score/class_precision.cc
	score/class_precision.h
	score/items_processed.cc
	score/items_processed.h
	score/perplexity.cc
	score/perplexity.h
	score/sparsity_phi.cc
	score/sparsity_phi.h
	score/sparsity_theta.cc
	score/sparsity_theta.h
	score/theta_snippet.cc
	score/theta_snippet.h
	score/top_tokens.cc
	score/top_tokens.h
	score/topic_kernel.cc
	score/topic_kernel.h
	score/topic_mass_phi.cc
	score/topic_mass_phi.h
	utility/blas.cc
	utility/blas.h
	utility/ifstream_or_cin.h
)

FILE(GLOB_RECURSE SRC_LIST_OTHER
	regularizer_interface.cc
	regularizer_interface.h
	score_calculator_interface.cc
	score_calculator_interface.h
	c_interface.cc
	c_interface.h
	messages.pb.cc
	messages.pb.h
)

FILE(GLOB_RECURSE SRC_LIST_CORE                core/*.cc                core/*.h                )
FILE(GLOB_RECURSE SRC_LIST_REGULARIZER         regularizer/*.cc         regularizer/*.h         )
FILE(GLOB_RECURSE SRC_LIST_SCORE               score/*.cc               score/*.h               )
FILE(GLOB_RECURSE SRC_LIST_UTILITY             utility/*.cc             utility/*.h             )

source_group("core"                FILES ${SRC_LIST_CORE}                )
source_group("regularizer"         FILES ${SRC_LIST_REGULARIZER}         )
source_group("score"               FILES ${SRC_LIST_SCORE}               )
source_group("utility"             FILES ${SRC_LIST_UTILITY}             )
source_group(""                    FILES ${SRC_LIST_OTHER}               )

add_definitions("-Dartm_EXPORTS")
add_library(artm-static STATIC ${SRC_LIST})


# Shared 'artm' library
set(SRC_LIST c_interface.cc c_interface.h)
source_group("" FILES c_interface.cc c_interface.h)

set(ARTM_LIB_DEPS  artm-static
                   gflags-static
                   ${Boost_LIBRARIES}
                   ${PROTOBUF_LIBRARIES}
                   ${GLOG_LIBRARIES})

add_library(artm SHARED ${SRC_LIST})

target_link_libraries(artm ${ARTM_LIB_DEPS})

if(MSVC)
  install(TARGETS artm DESTINATION bin)
  install(FILES messages.proto messages.pb.h messages.pb.cc c_interface.h cpp_interface.h cpp_interface.cc DESTINATION src)
else()
  install(TARGETS artm DESTINATION lib)
  # FIXME: install headers to /usr/include, protobuf and compiled messages --- to appropriate path
  #install(FILES messages.proto messages.pb.h messages.pb.cc c_interface.h cpp_interface.h cpp_interface.cc DESTINATION src)
endif()

