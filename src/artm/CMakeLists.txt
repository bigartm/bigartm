include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

# add custom command for creating from *.proto templates
add_custom_command(
  OUTPUT
  ${CMAKE_CURRENT_LIST_DIR}/messages.pb.cc
  ${CMAKE_CURRENT_LIST_DIR}/messages.pb.h
  ${CMAKE_CURRENT_LIST_DIR}/core/internals.pb.cc
  ${CMAKE_CURRENT_LIST_DIR}/core/internals.pb.h
  DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/messages.proto
  ${CMAKE_CURRENT_LIST_DIR}/core/internals.proto
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} artm/messages.proto --cpp_out=./
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} artm/core/internals.proto --cpp_out=./
  COMMENT "Generating headers and sources from *.proto files"
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..)

set(GENERATED_PROTO_FILES ${GENERATED_PROTO_FILES}
  ${CMAKE_CURRENT_LIST_DIR}/messages.pb.cc
  ${CMAKE_CURRENT_LIST_DIR}/messages.pb.h
  ${CMAKE_CURRENT_LIST_DIR}/core/internals.pb.cc
  ${CMAKE_CURRENT_LIST_DIR}/core/internals.pb.h)

set_source_files_properties(${GENERATED_PROTO_FILES} PROPERTIES GENERATED TRUE)

if (MSVC)
  set_source_files_properties(${GENERATED_PROTO_FILES} PROPERTIES COMPILE_FLAGS "/wd4244 /wd4267")
endif (MSVC)

# add custom target
# to bind targets with *.proto dependencies and
# *.proto generation
# as a result, we specify the order
add_custom_target(proto_generation ALL
  DEPENDS
  ${GENERATED_PROTO_FILES})

# The version number
set(ARTM_VERSION_MAJOR 0)
set(ARTM_VERSION_MINOR 9)
set(ARTM_VERSION_PATCH 0)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${CMAKE_CURRENT_LIST_DIR}/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/version.h"
  )

configure_file (
  "${CMAKE_CURRENT_LIST_DIR}/version.rc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/version.rc"
  @ONLY)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/utility/progress_printer.cc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/utility/progress_printer.cc"
  )

include_directories("${CMAKE_CURRENT_BINARY_DIR}/..")

# make separate objective files for every 'external' *.pb.cc file
# rationale - compiling every *.pb.cc file for different targets
# is very computationaly intensive
add_library(messages_proto STATIC EXCLUDE_FROM_ALL
    messages.pb.cc
    messages.pb.h)
add_dependencies(messages_proto proto_generation)

add_library(internals_proto STATIC EXCLUDE_FROM_ALL
    core/internals.pb.cc
    core/internals.pb.h)
add_dependencies(internals_proto proto_generation)

# Static 'artm' library

set(SRC_LIST
	c_interface.cc
	c_interface.h
	regularizer_interface.cc
	regularizer_interface.h
	score_calculator_interface.cc
	score_calculator_interface.h
	core/batch_manager.cc
	core/batch_manager.h
	core/cache_manager.cc
	core/cache_manager.h
	core/call_on_destruction.h
	core/check_messages.h
	core/collection_parser.cc
	core/collection_parser.h
	core/common.h
	core/cuckoo_watch.cc
	core/cuckoo_watch.h
	core/dense_phi_matrix.cc
	core/dense_phi_matrix.h
	core/dictionary.cc
	core/dictionary.h
	core/dictionary_operations.cc
	core/dictionary_operations.h
	core/exceptions.h
	core/helpers.cc
	core/helpers.h
	core/instance.cc
	core/instance.h
	core/master_component.cc
	core/master_component.h
	core/processor.cc
	core/processor.h
	core/processor_input.cc
	core/processor_input.h
	core/protobuf_helpers.h
	core/protobuf_serialization.h
	core/protobuf_serialization.cc
	core/phi_matrix.h
	core/phi_matrix_operations.cc
	core/phi_matrix_operations.h
	core/score_manager.cc
	core/score_manager.h
	core/template_manager.h
	core/thread_safe_holder.h
	core/token.cc
	core/token.h
	core/transform_function.h
	core/transform_function.cc
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
	regularizer/topic_selection_theta.cc
	regularizer/topic_selection_theta.h
	regularizer/biterms_phi.cc
	regularizer/biterms_phi.h
	regularizer/hierarchy_sparsing_theta.cc
	regularizer/hierarchy_sparsing_theta.h
	regularizer/topic_segmentation_ptdw.cc
	regularizer/topic_segmentation_ptdw.h
	regularizer/smooth_time_in_topics_phi.cc
	regularizer/smooth_time_in_topics_phi.h
	regularizer/net_plsa_phi.cc
	regularizer/net_plsa_phi.h
	score/class_precision.cc
	score/class_precision.h
	score/items_processed.cc
	score/items_processed.h
	score/peak_memory.cc
	score/peak_memory.h
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
	score/background_tokens_ratio.cc
	score/background_tokens_ratio.h
	utility/blas.cc
	utility/blas.h
	utility/ifstream_or_cin.h
	utility/memory_usage.h
	${CMAKE_CURRENT_BINARY_DIR}/utility/progress_printer.cc
	utility/progress_printer.h
)

FILE(GLOB_RECURSE SRC_LIST_OTHER
	regularizer_interface.cc
	regularizer_interface.h
	score_calculator_interface.cc
	score_calculator_interface.h
	c_interface.cc
	c_interface.h
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
target_link_libraries(artm-static messages_proto internals_proto)
add_dependencies(artm-static messages_proto internals_proto)


# Shared 'artm' library
set(SRC_LIST c_interface.cc c_interface.h)
source_group("" FILES c_interface.cc c_interface.h)

set(ARTM_LIB_DEPS  artm-static
                   gflags-static
                   ${Boost_LIBRARIES}
                   ${PROTOBUF_LIBRARIES}
                   ${GLOG_LIBRARIES})

add_library(artm SHARED ${SRC_LIST}
    ${CMAKE_CURRENT_BINARY_DIR}/version.rc)

target_link_libraries(artm ${ARTM_LIB_DEPS})

if (MSVC)
  install(TARGETS artm DESTINATION bin)
  install(FILES messages.proto messages.pb.h messages.pb.cc c_interface.h cpp_interface.h cpp_interface.cc DESTINATION src)
else (MSVC)
  install(TARGETS artm DESTINATION lib)
  # FIXME: install headers to /usr/include, protobuf and compiled messages --- to appropriate path
  #install(FILES messages.proto messages.pb.h messages.pb.cc c_interface.h cpp_interface.h cpp_interface.cc DESTINATION src)
endif (MSVC)
