if (WIN32)
  add_test(NAME python_wrapper_tests
    COMMAND py.test --junit-xml=${CMAKE_CURRENT_BINARY_DIR}/junit.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
else (WIN32)
  add_test(NAME python_wrapper_tests
    COMMAND py.test --forked --junit-xml=${CMAKE_CURRENT_BINARY_DIR}/junit.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif (WIN32)

set_tests_properties(python_wrapper_tests PROPERTIES
  ENVIRONMENT "ARTM_SHARED_LIBRARY=$<TARGET_FILE:artm>")
