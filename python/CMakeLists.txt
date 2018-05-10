# vim: set ts=2 sw=2:
project(python)


if (BUILD_INTERNAL_PYTHON_API)
  if (BUILD_TESTS)
    add_subdirectory(tests)
  endif (BUILD_TESTS)

  # perform "python setup.py build" during executing make
  add_custom_target(python_bigartm_build ALL
    ${PYTHON} ${CMAKE_CURRENT_LIST_DIR}/setup.py build --protoc_executable=$<TARGET_FILE:protoc>
    COMMENT "Building python package bigartm"
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  add_dependencies(python_bigartm_build protoc)
  add_custom_target(python_bigartm_wheel ALL
    ${CMAKE_COMMAND} -E env ARTM_SHARED_LIBRARY=$<TARGET_FILE:artm> ${PYTHON} setup.py bdist_wheel -d ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building wheel bigartm"
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  add_dependencies(python_bigartm_wheel python_bigartm_build artm)

  option(INSTALL_PYTHON_PACKAGE "Install python package as well during make install stage" OFF)

  if (WIN32)
    FILE(GLOB PYTHON_ARTM artm/*.py)
    install(FILES ${PYTHON_ARTM} DESTINATION python/artm)
    FILE(GLOB PYTHON_ARTM artm/wrapper/*.py)
    install(FILES ${PYTHON_ARTM} DESTINATION python/artm/wrapper)
    # This file is generated, therefore needs to be specified without *
    install(FILES artm/wrapper/messages_pb2.py DESTINATION python/artm/wrapper)
    FILE(GLOB PYTHON_EXAMPLES tests/wrapper/*.py)
    install(FILES ${PYTHON_EXAMPLES} DESTINATION python/tests/wrapper)
    install(FILES setup.py DESTINATION python)
  elseif (UNIX)
    if (INSTALL_PYTHON_PACKAGE)
      install(CODE "message(\"Installing python package bigartm\")")
      install(CODE "execute_process(COMMAND ${PYTHON} setup.py install
                                    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})")
    endif (INSTALL_PYTHON_PACKAGE)
  endif (WIN32)
endif (BUILD_INTERNAL_PYTHON_API)

