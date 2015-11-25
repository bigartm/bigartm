project(bigartm)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	srcmain.cc
	${CMAKE_SOURCE_DIR}/src/artm/cpp_interface.cc
	${CMAKE_SOURCE_DIR}/src/artm/messages.pb.cc
)

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC ON)
find_package(Boost COMPONENTS REQUIRED ${BIGARTM_BOOST_COMPONENTS})

add_executable(bigartm ${SRC_LIST})

option(BUILD_STATIC_BIGARTM "Request build of static executable bigartm" ON)

if (WIN32)
  target_link_libraries(bigartm artm ${Boost_LIBRARIES})
else (WIN32)
  if (BUILD_STATIC_BIGARTM)
    set(CMAKE_EXE_LINKER_FLAGS "-static")
  endif (BUILD_STATIC_BIGARTM)
  target_link_libraries(bigartm
	  artm-static
	  ${PROTOBUF_LIBRARIES}
	  ${GLOG_LIBRARIES}
	  ${Boost_LIBRARIES})
endif (WIN32)

install(TARGETS bigartm DESTINATION bin)
