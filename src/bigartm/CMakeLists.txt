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

target_link_libraries(bigartm ${BOOST_IMPORTED_TARGETS})

install(TARGETS bigartm DESTINATION bin)
