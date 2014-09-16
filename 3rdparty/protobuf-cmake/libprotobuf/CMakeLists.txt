project( libprotobuf )

include( ../protobuf-common.cmake )

append_if_exist(SOURCES
  ${PROTOBUF_ROOT}/src/google/protobuf/compiler/importer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/compiler/parser.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor_database.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/dynamic_message.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/extension_set.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/extension_set_heavy.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_reflection.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_util.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/coded_stream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/gzip_stream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/printer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/tokenizer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream_impl.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream_impl_lite.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/message.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/message_lite.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/reflection_ops.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/repeated_field.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/service.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/atomicops_internals_x86_gcc.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/atomicops_internals_x86_msvc.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/common.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/once.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/stringprintf.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/structurally_valid.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/strutil.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/substitute.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/text_format.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/unknown_field_set.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/wire_format.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/wire_format_lite.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/hash.cc
)

add_library(libprotobuf STATIC ${SOURCES} )

set_target_properties(libprotobuf PROPERTIES OUTPUT_NAME "protobuf")
set(PROTOBUF_LIBRARY libprotobuf CACHE INTERNAL "protobuf library")

if(MSVC)
# Install python files
append_if_exist(INSTALL_FILES_LIST
	${PROTOBUF_ROOT}/src/google/protobuf/compiler/plugin.proto
	${PROTOBUF_ROOT}/src/google/protobuf/descriptor.proto
	${PROTOBUF_ROOT}/Python/ez_setup.py
	${PROTOBUF_ROOT}/Python/mox.py
	${PROTOBUF_ROOT}/Python/README.txt
	${PROTOBUF_ROOT}/Python/setup.py
	${PROTOBUF_ROOT}/Python/stubout.py
	${PROTOBUF_ROOT}/Python/google/__init__.py
	${PROTOBUF_ROOT}/Python/google/protobuf/descriptor.py
	${PROTOBUF_ROOT}/Python/google/protobuf/descriptor_database.py
	${PROTOBUF_ROOT}/Python/google/protobuf/descriptor_pool.py
	${PROTOBUF_ROOT}/Python/google/protobuf/message.py
	${PROTOBUF_ROOT}/Python/google/protobuf/message_factory.py
	${PROTOBUF_ROOT}/Python/google/protobuf/reflection.py
	${PROTOBUF_ROOT}/Python/google/protobuf/service.py
	${PROTOBUF_ROOT}/Python/google/protobuf/service_reflection.py
	${PROTOBUF_ROOT}/Python/google/protobuf/text_format.py
	${PROTOBUF_ROOT}/Python/google/protobuf/__init__.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/api_implementation.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/containers.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/cpp_message.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/decoder.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/descriptor_database_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/descriptor_pool_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/descriptor_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/encoder.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/enum_type_wrapper.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/factory_test1.proto
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/factory_test2.proto
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/generator_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/message_cpp_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/message_factory_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/message_listener.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/message_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/more_extensions.proto
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/more_extensions_dynamic.proto
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/more_messages.proto
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/python_message.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/reflection_cpp_generated_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/reflection_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/service_reflection_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/test_bad_identifiers.proto
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/test_util.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/text_format_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/type_checkers.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/unknown_fields_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/wire_format.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/wire_format_test.py
	${PROTOBUF_ROOT}/Python/google/protobuf/internal/__init__.py
	${PROTOBUF_ROOT}/Python/google/protobuf/pyext/python-proto2.cc
	${PROTOBUF_ROOT}/Python/google/protobuf/pyext/python_descriptor.cc
	${PROTOBUF_ROOT}/Python/google/protobuf/pyext/python_descriptor.h
	${PROTOBUF_ROOT}/Python/google/protobuf/pyext/python_protobuf.cc
	${PROTOBUF_ROOT}/Python/google/protobuf/pyext/python_protobuf.h
    )

foreach(fil ${INSTALL_FILES_LIST})
    file(RELATIVE_PATH relative_file ${PROTOBUF_ROOT} ${fil})

    get_filename_component(relative_header_dir ${relative_file} PATH)

    install(FILES
        ${fil}
        DESTINATION ${INCLUDE_DIR}/../protobuf/${relative_header_dir}
        )
endforeach()
endif()  # if(MSVC)

set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_ROOT}/src CACHE INTERNAL "protobuf library")
