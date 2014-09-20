include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

# Static 'artm' library

set(SRC_LIST
	messages.pb.cc
	messages.pb.h
	regularizer_interface.cc
	regularizer_interface.h
	score_calculator_interface.cc
	score_calculator_interface.h
	core/batch_manager.cc
	core/batch_manager.h
	core/call_on_destruction.h
	core/collection_parser.cc
	core/collection_parser.h
	core/common.h
	core/data_loader.cc
	core/data_loader.h
	core/dictionary.h
	core/exceptions.h
	core/generation.cc
	core/generation.h
	core/helpers.cc
	core/helpers.h
	core/instance.cc
	core/instance.h
	core/instance_schema.cc
	core/instance_schema.h
	core/internals.pb.cc
	core/internals.pb.h
	core/internals.rpcz.cc
	core/internals.rpcz.h
	core/master_component.cc
	core/master_component.h
	core/master_component_service_impl.cc
	core/master_component_service_impl.h
	core/master_interface.h
	core/master_proxy.cc
	core/master_proxy.h
	core/merger.cc
	core/merger.h
	core/node_controller.cc
	core/node_controller.h
	core/node_controller_service_impl.cc
	core/node_controller_service_impl.h
	core/processor.cc
	core/processor.h
	core/protobuf_helpers.h
	core/template_manager.h
	core/thread_safe_holder.h
	core/topic_model.cc
	core/topic_model.h
	core/zmq_context.h
	regularizer_sandbox/decorrelator_phi.cc
	regularizer_sandbox/decorrelator_phi.h
	regularizer_sandbox/dirichlet_phi.cc
	regularizer_sandbox/dirichlet_phi.h
	regularizer_sandbox/dirichlet_theta.cc
	regularizer_sandbox/dirichlet_theta.h
	regularizer_sandbox/multilanguage_phi.cc
	regularizer_sandbox/multilanguage_phi.h
	regularizer_sandbox/smooth_sparse_phi.cc
	regularizer_sandbox/smooth_sparse_phi.h
	regularizer_sandbox/smooth_sparse_theta.cc
	regularizer_sandbox/smooth_sparse_theta.h
	score_sandbox/items_processed.cc
	score_sandbox/items_processed.h
	score_sandbox/perplexity.cc
	score_sandbox/perplexity.h
	score_sandbox/sparsity_phi.cc
	score_sandbox/sparsity_phi.h
	score_sandbox/sparsity_theta.cc
	score_sandbox/sparsity_theta.h
	score_sandbox/theta_snippet.cc
	score_sandbox/theta_snippet.h
	score_sandbox/top_tokens.cc
	score_sandbox/top_tokens.h
	score_sandbox/topic_kernel.cc
	score_sandbox/topic_kernel.h
)

FILE(GLOB_RECURSE SRC_LIST_OTHER
	regularizer_interface.cc
	regularizer_interface.h
	score_calculator_interface.cc
	score_calculator_interface.h
	messages.pb.cc
	messages.pb.h
)

FILE(GLOB_RECURSE SRC_LIST_CORE                core/*.cc                core/*.h                )
FILE(GLOB_RECURSE SRC_LIST_REGULARIZER         regularizer/*.cc         regularizer/*.h         )
FILE(GLOB_RECURSE SRC_LIST_REGULARIZER_SANDBOX regularizer_sandbox/*.cc regularizer_sandbox/*.h )
FILE(GLOB_RECURSE SRC_LIST_SCORE               score/*.cc               score/*.h               )
FILE(GLOB_RECURSE SRC_LIST_SCORE_SANDBOX       score_sandbox/*.cc       score_sandbox/*.h       )

source_group("core"                FILES ${SRC_LIST_CORE}                )
source_group("regularizer"         FILES ${SRC_LIST_REGULARIZER}         )
source_group("regularizer_sandbox" FILES ${SRC_LIST_REGULARIZER_SANDBOX} )
source_group("score"               FILES ${SRC_LIST_SCORE}               )
source_group("score_sandbox"       FILES ${SRC_LIST_SCORE_SANDBOX}       )
source_group(""                    FILES ${SRC_LIST_OTHER}               )

add_definitions("-Dartm_EXPORTS")
add_library(artm-static STATIC ${SRC_LIST})


# Shared 'artm' library
set(SRC_LIST c_interface.cc c_interface.h)
source_group("" FILES c_interface.cc c_interface.h)

set(ARTM_LIB_DEPS  artm-static
		   gflags-static
                   rpcz_static
                   ${Boost_THREAD_LIBRARIES}
                   ${Boost_SYSTEM_LIBRARIES}
                   ${Boost_FILESYSTEM_LIBRARIES}
                   ${Boost_IOSTREAMS_LIBRARIES}
                   ${PROTOBUF_LIBRARIES}
                   ${GLOG_LIBRARIES}
                   ${ZEROMQ_LIBRARIES})

add_library(artm SHARED ${SRC_LIST})

target_link_libraries(artm ${ARTM_LIB_DEPS})

install(TARGETS artm DESTINATION bin)
install(FILES messages.proto messages.pb.h messages.pb.cc c_interface.h cpp_interface.h cpp_interface.cc DESTINATION src)
