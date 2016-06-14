project(python)

add_custom_target(python_generate_proto ALL
  python ${CMAKE_CURRENT_LIST_DIR}/gen_proto.py --protoc_executable=$<TARGET_FILE:protoc>
  COMMENT "Generating python protobuf files"
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
add_dependencies(python_generate_proto protoc)

# perform "python setup.py build" during executing make
if (BUILD_INTERNAL_PYTHON_API)
  add_custom_target(python_bigartm_build ALL
    python ${CMAKE_CURRENT_LIST_DIR}/setup.py build
    COMMENT "Building python package bigartm"
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  add_dependencies(python_bigartm_build python_generate_proto artm)

  if (MSVC)
    FILE(GLOB PYTHON_ARTM artm/*.py)
    install(FILES ${PYTHON_ARTM} DESTINATION python/artm)
    FILE(GLOB PYTHON_ARTM artm/wrapper/*.py)
    install(FILES ${PYTHON_ARTM} DESTINATION python/artm/wrapper)
    FILE(GLOB PYTHON_EXAMPLES tests/wrapper/*.py)
    install(FILES ${PYTHON_EXAMPLES} DESTINATION python/tests/wrapper)
    install(FILES setup.py DESTINATION python)
    install(FILES ../datasets/docword.kos.txt ../datasets/vocab.kos.txt DESTINATION python/examples)
  elseif (UNIX)
    install(CODE "message(\"Installing python package bigartm\")")
    install(CODE "execute_process(COMMAND python setup.py install
      WORKING_DIRECTORY \"${CMAKE_CURRENT_LIST_DIR}\")")
  endif (MSVC)
endif (BUILD_INTERNAL_PYTHON_API)

