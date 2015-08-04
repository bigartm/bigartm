project(python)

if(MSVC)
  FILE(GLOB PYTHON_ARTM artm/*.py)
  install(FILES ${PYTHON_ARTM} DESTINATION python/artm)

  FILE(GLOB PYTHON_EXAMPLES examples/*.py)
  install(FILES ${PYTHON_EXAMPLES} DESTINATION python/examples)

  FILE(GLOB PYTHON_EXPERIMENTS experiments/*.py)
  install(FILES ${PYTHON_EXPERIMENTS} DESTINATION python/experiments)

  install(FILES ../datasets/docword.kos.txt ../datasets/vocab.kos.txt DESTINATION python/examples)
endif()
