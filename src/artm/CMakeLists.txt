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
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} artm/messages.proto --cpp_out=dllexport_decl=DLL_PUBLIC:./
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

target_link_libraries(internals_proto libprotobuf)
target_link_libraries(messages_proto libprotobuf)
target_compile_definitions(messages_proto PRIVATE ARTM_STATIC_DEFINE)
if (MSVC)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/messages.pb.cc
        PROPERTIES COMPILE_FLAGS "/FI\"${CMAKE_CURRENT_BINARY_DIR}/artm_export.h\" /wd4244 /wd4267")
else (MSVC)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/messages.pb.cc
        PROPERTIES COMPILE_FLAGS "-include artm_export.h")
endif (MSVC)

# Static 'artm' library

set(SRC_LIST
	c_interface.cc
	c_interface.h
	cpp_interface.cc
	cpp_interface.h
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
	core/cooccurrence_collector.cc
	core/cooccurrence_collector.h
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
	core/processor_helpers.cc
	core/processor_helpers.h
	core/processor_transaction_helpers.cc
	core/processor_transaction_helpers.h
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
	cpp_interface.cc
	cpp_interface.h
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

add_library(artm-static STATIC ${SRC_LIST})
target_link_libraries(artm-static
    PUBLIC messages_proto internals_proto glog)
add_dependencies(artm-static messages_proto internals_proto)
target_compile_definitions(artm-static PRIVATE ARTM_STATIC_DEFINE)

if (WIN32)
    # This library is needed for GetProcessMemoryInfo and is not linked by default on MinGW
    target_link_libraries(artm-static PUBLIC psapi)
endif (WIN32)

target_include_directories(artm-static PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(artm-static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>)

# Shared 'artm' library
# We need to rebuild these .cc files in order to get dllexports
set(SRC_LIST
		c_interface.h
		c_interface.cc
		cpp_interface.h
		cpp_interface.cc
		messages.pb.h
		messages.pb.cc)
source_group("" FILES ${SRC_LIST})
add_library(artm SHARED ${SRC_LIST}
    ${CMAKE_CURRENT_BINARY_DIR}/version.rc)

# Make protobuf export symbols from our shared library
target_compile_definitions(artm
    PUBLIC PROTOBUF_USE_DLLS
    PRIVATE LIBPROTOBUF_EXPORTS)

if (WIN32)
    set_target_properties(artm PROPERTIES PREFIX "")
endif (WIN32)

# Using PRIVATE here so that consumers of the library do not link explicitly
# to boost, etc. This should be handled by the linker via DT_NEEDED or similar
# But we have to repeat include dir here for artm_export.h
target_link_libraries(artm PRIVATE artm-static ${BOOST_IMPORTED_TARGETS})
target_include_directories(artm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>)
target_include_directories(artm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
# Support compiling with bundled protobuf headers from build-tree
target_include_directories(artm PUBLIC $<BUILD_INTERFACE:${PROTOBUF_INCLUDE_DIR}>)

include(GenerateExportHeader)
generate_export_header(artm
		EXPORT_MACRO_NAME DLL_PUBLIC)

install(TARGETS artm EXPORT artm
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include/artm
    )
install(FILES
    c_interface.h
    cpp_interface.h
    messages.pb.h
    messages.proto
    ${CMAKE_CURRENT_BINARY_DIR}/artm_export.h
    DESTINATION include/artm
    )

export(EXPORT artm
    FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/artm-targets.cmake"
    NAMESPACE artm::
    )
configure_file(cmake/artm-config.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/artm-config.cmake"
    COPYONLY
    )

set(ConfigPackageLocation lib/cmake/artm)
install(EXPORT artm
    FILE artm-targets.cmake
    NAMESPACE artm::
    DESTINATION ${ConfigPackageLocation}
    )
install(FILES cmake/artm-config.cmake
    DESTINATION ${ConfigPackageLocation}
    )

# This ugly hack calls cmake install script for protobuf directly and
# installs only its headers
install(CODE
    "
    set(COMPONENT_BACKUP \${CMAKE_INSTALL_COMPONENT})
    set(CMAKE_INSTALL_COMPONENT \"protobuf-headers\")
    include(${PROTOBUF_BINARY_DIR}/cmake_install.cmake)
    set(CMAKE_INSTALL_COMPONENT \"protobuf-protos\")
    include(${PROTOBUF_BINARY_DIR}/cmake_install.cmake)
    if (WIN32)
        set(CMAKE_INSTALL_COMPONENT \"protoc\")
        include(${PROTOBUF_BINARY_DIR}/cmake_install.cmake)
    endif (WIN32)
    set(CMAKE_INSTALL_COMPONENT \${COMPONENT_BACKUP})
    "
    )
