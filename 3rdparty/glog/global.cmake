##################################################
#
#	BUILD/GLOBAL.CMAKE
#
# 	This file is for providing a defined environment
#	of compiler definitions/macros and cmake functions
#	or variables throughout several projects. It can
#	be included twice or more without any issues and
#   will automatically included the utility files 
#   compiler.cmake and macros.cmake
#
#	(c) 2009-2012 Marius Zwicker
#
##################################################

### CONFIGURATION SECTION

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# CMAKE_CURRENT_LIST_DIR is available after CMake 2.8.3 only
# but we support 2.8.0 as well
if( NOT CMAKE_CURRENT_LIST_DIR )
    string(REPLACE "/global.cmake" "" CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}")
endif()

# path to the mz tools files
set(MZ_TOOLS_PATH "${CMAKE_CURRENT_LIST_DIR}")

### END OF CONFIGURATION SECTION

# BOF: global.cmake
if(NOT HAS_MZ_GLOBAL)
	set(HAS_MZ_GLOBAL true)

  # detect compiler
  include("${MZ_TOOLS_PATH}/compiler.cmake")

  # user info
  message("-- configuring for build type: ${CMAKE_BUILD_TYPE}")
  
  # macros
  include("${MZ_TOOLS_PATH}/macros.cmake")

# EOF: global.cmake
endif() 
