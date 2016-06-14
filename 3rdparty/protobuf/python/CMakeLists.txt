if (UNIX)
  # perform "python setup.py build" during executing make
  add_custom_target(python_protobuf_build ALL
    python ${CMAKE_CURRENT_LIST_DIR}/setup.py build
    COMMENT "Building python package protobuf 2.5.1-pre"
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  add_dependencies(python_protobuf_build protoc)
  install(CODE "message(\"Installing python package protobuf\")")
  install(CODE "execute_process(COMMAND python setup.py install
    WORKING_DIRECTORY \"${CMAKE_CURRENT_LIST_DIR}\")")
endif (UNIX)

