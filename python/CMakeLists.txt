project(python)

if(MSVC)
  FILE(GLOB PYTHON_ARTM artm/*.py)
  install(FILES ${PYTHON_ARTM} DESTINATION python/artm)

  FILE(GLOB PYTHON_ARTM artm/wrapper/*.py)
  install(FILES ${PYTHON_ARTM} DESTINATION python/artm/wrapper)

  FILE(GLOB PYTHON_EXAMPLES tests/wrapper/*.py)
  install(FILES ${PYTHON_EXAMPLES} DESTINATION python/tests/wrapper)

  install(FILES setup.py DESTINATION python)

  install(FILES ../datasets/docword.kos.txt ../datasets/vocab.kos.txt DESTINATION python/examples)
elseif(UNIX)
  # perform "python setup.py build" during executing make
  add_custom_target(python_build ALL
    echo "Building python package bigartm"
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/setup.py build
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  install(CODE "message(\"Installing python package bigartm\")")
  install(CODE "execute_process(COMMAND python setup.py install\
    WORKING_DIRECTORY \"${CMAKE_CURRENT_LIST_DIR}\")")
endif(MSVC)
