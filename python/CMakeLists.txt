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
endif()
