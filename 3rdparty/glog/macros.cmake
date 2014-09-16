########################################################################
#
#	BUILD/MACROS.CMAKE
#
# 	This file provides some useful macros to
#	simplify adding of componenents and other
#	taskss
#	(c) 2009-2012 Marius Zwicker
#
# This file defines a whole bunch of macros
# to add a subdirectory containing another
# CMakeLists.txt as "Subproject". All these
# Macros are not doing that much but giving
# feedback to tell what kind of component was
# added. In all cases NAME is the name of your
# subproject and FOLDER is a relative path to
# the folder containing a CMakeLists.txt
#
# mz_add_library <NAME> <FOLDER>
#		macro for adding a new library
#
# mz_add_executable <NAME> <FOLDER>
# 		macro for adding a new executable
#
# mz_add_control <NAME> <FOLDER>
#		macro for adding a new control
#
# mz_add_testtool <NAME> <FOLDER>
#		macro for adding a folder containing testtools
#
# mz_add_external <NAME> <FOLDER>
#		macro for adding an external library/tool dependancy
#
# mz_target_props <target>
#		automatically add a "D" postfix when compiling in debug
#       mode to the given target
#
# mz_auto_moc <mocced> ...
#		search all passed files in (...) for Q_OBJECT and if found
#		run moc on them via qt4_wrap_cpp. Assign the output files
#		to <mocced>. Improves the version provided by cmake by searching
#       for Q_OBJECT first and thus reducing the needed calls to moc
#
# mz_find_include_library <name>  SYS <version> SRC <directory> <include_dir> <target>
#       useful when providing a version of a library within the
#       own sourcetree but prefer the system's library version over it.
#       Will search for the given header in the system includes and when
#       not found, it will include the given directory which should contain
#       a cmake file defining the given target.
#       After calling this macro the following variables will be declared:
#           <name>_INCLUDE_DIRS The directory containing the header or 
#                              the passed include_dir if the lib was not 
#                              found on the system
#           <name>_LIBRARIES The libs to link against - either lib or target
#           <name>_FOUND true if the lib was found on the system
#
########################################################################

# if global.cmake was not included yet, report it
if (NOT HAS_MZ_GLOBAL)
	message(FATAL_ERROR "!! include global.cmake before including this file !!")
endif()

########################################################################
## no need to change anything beyond here
########################################################################

macro(mz_add_library NAME FOLDER)
	mz_message("adding library ${NAME}")
	__mz_add_target(${NAME} ${FOLDER})
endmacro()

macro(mz_add_executable NAME FOLDER)
	mz_message("adding executable ${NAME}")
	__mz_add_target(${NAME} ${FOLDER})
endmacro()

macro(mz_add_control NAME FOLDER)
	mz_message("adding control ${NAME}")
	__mz_add_target(${NAME} ${FOLDER})
endmacro()

macro(mz_add_testtool NAME FOLDER)
	mz_message("adding testtool ${NAME}")
	__mz_add_target(${NAME} ${FOLDER})
endmacro()

macro(mz_add_external NAME FOLDER)
	mz_message("adding external dependancy ${NAME}")
	__mz_add_target(${NAME} ${FOLDER})
endmacro()

macro(__mz_add_target NAME FOLDER)
    add_subdirectory(${FOLDER})
endmacro()

macro(mz_target_props NAME)
    set_target_properties(${NAME} PROPERTIES DEBUG_POSTFIX "D")
endmacro()

macro(__mz_extract_files _qt_files)
	set(${_qt_files})
	FOREACH(_current ${ARGN})
		file(STRINGS ${_current} _content LIMIT_COUNT 1 REGEX .*Q_OBJECT.*)
		if("${_content}" MATCHES .*Q_OBJECT.*)
			LIST(APPEND ${_qt_files} "${_current}")
		endif()
	ENDFOREACH(_current)
endmacro()

macro(mz_auto_moc mocced)
	#mz_debug_message("mz_auto_moc input: ${ARGN}")
	
	set(_mocced "")
	# determine the required files
	__mz_extract_files(to_moc ${ARGN})
	#mz_debug_message("mz_auto_moc mocced: ${to_moc}")
	qt4_wrap_cpp(_mocced ${to_moc})
	set(${mocced} ${${mocced}} ${_mocced})
endmacro()

include(CheckIncludeFiles)

if( NOT CMAKE_MODULE_PATH )
    set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/Modules" )
endif()

macro(mz_check_include_files FILE VAR)
	if( IOS )
		mz_debug_message("Using custom check_include_files")
		
		if( NOT DEFINED FOUND_${VAR} )
			mz_message("Looking for include files ${FILE}")
			find_file( ${VAR} 
				NAMES ${FILE} 
				PATHS ${CMAKE_REQUIRED_INCLUDES}
			)
			if( ${VAR} )
				mz_message("Looking for include files ${FILE} - found")
				set( FOUND_${VAR} ${${VAR}} CACHE INTERNAL FOUND_${VAR} )
			else()
				mz_message("Looking for include files ${FILE} - not found")
			endif()
		else()
			set( ${VAR} ${FOUND_${VAR}} )
		endif()
		
	else()
		mz_debug_message("Using native check_include_files")
		
		CHECK_INCLUDE_FILES( ${FILE} ${VAR} )
	endif()
endmacro()

macro(mz_find_include_library _NAME SYS _VERSION SRC _DIRECTORY _INC_DIR _TARGET)
    
    STRING(TOUPPER ${_NAME} _NAME_UPPER)
    
    find_package( ${_NAME} )
    if( NOT ${_NAME_UPPER}_FOUND )
        set(${_NAME_UPPER}_INCLUDE_DIRS ${_INC_DIR})
        set(${_NAME_UPPER}_LIBRARIES ${_TARGET})
        
        mz_add_library(${_NAME} ${_DIRECTORY})    
    endif()

endmacro()

