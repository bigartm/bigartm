if(UNIX)
  # perform "python setup.py build" during executing make
  add_custom_target(python_protobuf_build ALL
    echo "Building python package protobuf 2.5.1-pre"
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/setup.py build
    DEPENDS protoc
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  install(CODE "message(\"Installing python package protobuf\")")
  install(CODE "execute_process(COMMAND python setup.py install\
    WORKING_DIRECTORY \"${CMAKE_CURRENT_LIST_DIR}\")")
endif(UNIX)

