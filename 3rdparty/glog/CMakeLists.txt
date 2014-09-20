cmake_minimum_required(VERSION 2.8)
INCLUDE (CheckIncludeFiles)
INCLUDE (CheckFunctionExists)
INCLUDE (CheckSymbolExists)
project(google-glog)

if(NOT HAS_MZ_GLOBAL)
    include(global.cmake)
endif()

CHECK_INCLUDE_FILES (dlfcn.h HAVE_DLFCN_H)
CHECK_INCLUDE_FILES (execinfo.h HAVE_EXECINFO_H)
CHECK_INCLUDE_FILES (glob.h HAVE_GLOB_H)
CHECK_INCLUDE_FILES (inttypes.h HAVE_INTTYPES_H)
CHECK_INCLUDE_FILES (pthread.h HAVE_LIBPTHREAD)
CHECK_INCLUDE_FILES (libunwind.h HAVE_LIBUNWIND_H)
CHECK_INCLUDE_FILES (libunwind.h HAVE_LIB_UNWIND)
CHECK_INCLUDE_FILES (gflags/gflags.h HAVE_LIB_GFLAGS)
set(HAVE_LIB_GFLAGS TRUE) # we can be sure as this is a build dependency
CHECK_INCLUDE_FILES (memory.h HAVE_MEMORY_H)
CHECK_INCLUDE_FILES (pthread.h HAVE_PTHREAD_H)
CHECK_INCLUDE_FILES (pwd.h HAVE_PWD_H)
CHECK_INCLUDE_FILES (stdint.h HAVE_STDINT_H)
CHECK_INCLUDE_FILES (stdlib.h HAVE_STDLIB_H)
CHECK_INCLUDE_FILES (strings.h HAVE_STRINGS_H)
CHECK_INCLUDE_FILES (syscall.h HAVE_SYSCALL_H)
CHECK_INCLUDE_FILES (syslog.h HAVE_SYSLOG_H)
CHECK_INCLUDE_FILES (sys/stat.h HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILES (sys/call.h HAVE_SYS_CALL_H)
CHECK_INCLUDE_FILES (sys/time.h HAVE_SYS_TIME_H)
CHECK_INCLUDE_FILES (sys/types.h HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILES (sys/syscall.h HAVE_SYS_SYSCALL_H)
CHECK_INCLUDE_FILES (sys/ucontext.h HAVE_SYS_UCONTEXT_H)
CHECK_INCLUDE_FILES (sys/utsname.h HAVE_SYS_UTSNAME_H)
CHECK_INCLUDE_FILES (ucontext.h HAVE_UCONTEXT_H)
CHECK_INCLUDE_FILES (unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILES (unwind.h HAVE_UNWIND_H)

CHECK_FUNCTION_EXISTS(fcntl HAVE_FCNTL)
CHECK_FUNCTION_EXISTS(pthread_rwlock_rdlock HAVE_RWLOCK)
CHECK_FUNCTION_EXISTS(sigaltstack HAVE_SIGALTSTACK)
CHECK_FUNCTION_EXISTS(__builtin_expect HAVE___BUILTIN_EXPECT)
CHECK_FUNCTION_EXISTS(__sync_val_compare_and_swap HAVE___SYNC_VAL_COMPARE_AND_SWAP)

CONFIGURE_FILE(src/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/src/glog-config.h)

mz_use_default_compiler_settings()
mz_add_flag(GCC -fPIC)

if(WIN32)
	set(SRCS
		src/logging.cc
		src/windows/port.cc
		src/raw_logging.cc
		src/utilities.cc
		src/vlog_is_on.cc
	)
	
	set(HDRS
		src/base/commandlineflags.h
		src/windows/config.h
		src/base/googleinit.h
		src/windows/glog/log_severity.h
		src/windows/glog/logging.h
		src/base/mutex.h
		src/windows/port.h
		src/windows/glog/raw_logging.h
		src/windows/glog/stl_logging.h
		src/utilities.h
		src/windows/glog/vlog_is_on.h
	)

	include_directories(
		${CMAKE_CURRENT_SOURCE_DIR}/src/windows/
		${CMAKE_CURRENT_BINARY_DIR}/src/
		${GFLAGS_INCLUDE_DIRS}
	)

	mz_add_definition(GLOG_NO_ABBREVIATED_SEVERITIES)
else()
	set(SRCS
            src/logging.cc
            src/raw_logging.cc
            src/vlog_is_on.cc
            src/utilities.cc
            src/utilities.h
            src/demangle.cc src/demangle.h
            src/stacktrace.h
            src/stacktrace_generic-inl.h
            src/stacktrace_libunwind-inl.h
            src/stacktrace_powerpc-inl.h
            src/stacktrace_x86-inl.h
            src/stacktrace_x86_64-inl.h
            src/symbolize.cc src/symbolize.h
            src/signalhandler.cc
            src/base/mutex.h src/base/googleinit.h
            src/base/commandlineflags.h src/googletest.h
	)
	
	set(HDRS
		src/glog/log_severity.h
		src/glog/logging.h 
		src/glog/raw_logging.h 
		src/glog/vlog_is_on.h 
		src/glog/stl_logging.h
	)

	include_directories(
		${CMAKE_CURRENT_SOURCE_DIR}/src/
		${CMAKE_CURRENT_BINARY_DIR}/src/		
		${GFLAGS_INCLUDE_DIRS}
	)

	mz_add_definition(NO_THREADS)
	set(GLOG_3LIBS pthread)
        if( HAVE_LIB_UNWIND )
            set(GLOG_3LIBS ${GLOG_3LIBS} unwind)
        endif()
endif()

mz_add_flag(GCC -Wall -Wwrite-strings -Woverloaded-virtual -Wno-sign-compare)
add_library(google-glog STATIC ${SRCS} ${HDRS})
target_link_libraries(google-glog ${GFLAGS_LIBRARY} ${GLOG_3LIBS})
set_property(TARGET google-glog PROPERTY VERSION 0.3.3)
set_property(TARGET google-glog PROPERTY SOVERSION 0.3.3)
mz_target_props(google-glog)

if(0)  # disable install

if(WIN32)
    install(TARGETS google-glog
                LIBRARY DESTINATION lib
                RUNTIME DESTINATION bin
                ARCHIVE DESTINATION lib)
    install(DIRECTORY src/windows/glog
                DESTINATION include
                PATTERN "*.svn" EXCLUDE)
    install(DIRECTORY doc
                DESTINATION doc
                PATTERN "*.svn" EXCLUDE)
else()
    install(TARGETS google-glog
                LIBRARY DESTINATION lib
                RUNTIME DESTINATION bin
                ARCHIVE DESTINATION lib)
    install(DIRECTORY src/glog
                DESTINATION include
                PATTERN "*.svn" EXCLUDE)
    if(APPLE)
        install(DIRECTORY doc
                DESTINATION Library/Documentation/google-glog
                PATTERN "*.svn" EXCLUDE)
    else()
        install(DIRECTORY doc
                DESTINATION share/doc/google-glog-doc
                PATTERN "*.svn" EXCLUDE)
    endif()
endif()

endif()