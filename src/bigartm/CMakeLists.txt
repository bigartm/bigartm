project(bigartm)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	srcmain.cc
)

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC ON)
find_package(Boost COMPONENTS REQUIRED ${BIGARTM_BOOST_COMPONENTS})

add_executable(bigartm ${SRC_LIST})
add_dependencies(bigartm messages_proto)

if (BUILD_BIGARTM_CLI_STATIC)
  set(CMAKE_EXE_LINKER_FLAGS "-static")
endif (BUILD_BIGARTM_CLI_STATIC)

target_include_directories(bigartm PRIVATE $<TARGET_PROPERTY:glog,INTERFACE_INCLUDE_DIRECTORIES>)

if (WIN32)
  # FIXME find a way to export all needed symbols from protobuf without needing to link to it here
  target_link_libraries(bigartm artm ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES})
else (WIN32)
  target_link_libraries(bigartm artm-static)
endif (WIN32)

install(TARGETS bigartm DESTINATION bin)
