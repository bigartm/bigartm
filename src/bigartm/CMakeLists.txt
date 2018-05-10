project(bigartm)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	srcmain.cc
)

add_executable(bigartm ${SRC_LIST})

if (BUILD_BIGARTM_CLI_STATIC)
  set(CMAKE_EXE_LINKER_FLAGS "-static")
endif (BUILD_BIGARTM_CLI_STATIC)

target_include_directories(bigartm PRIVATE $<TARGET_PROPERTY:glog,INTERFACE_INCLUDE_DIRECTORIES>)

if (WIN32)
  target_link_libraries(bigartm artm)
else (WIN32)
  target_link_libraries(bigartm artm-static)
endif (WIN32)

if (NOT WIN32 AND BUILD_BIGARTM_CLI_STATIC)
  # To use static boost libraries, we need to re-find them
  unset(Boost_LIBRARIES)
  set(Boost_USE_STATIC_LIBS ON)
  find_package(Boost REQUIRED COMPONENTS ${BIGARTM_BOOST_COMPONENTS})
  target_include_directories(bigartm PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(bigartm ${Boost_LIBRARIES})
else (NOT WIN32 AND BUILD_BIGARTM_CLI_STATIC)
  # If we don't want static linkage, we can just use new approach with
  # imported targets like everywhere else in the project
  target_link_libraries(bigartm ${BOOST_IMPORTED_TARGETS})
endif (NOT WIN32 AND BUILD_BIGARTM_CLI_STATIC)

install(TARGETS bigartm DESTINATION bin)
