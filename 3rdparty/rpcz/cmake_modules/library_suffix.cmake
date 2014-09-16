# Enumerate the library suffix, needed for some of the variables coming out
# of the external projects that are not CMake.

if(UNIX)
  if(APPLE)
    set(link_library_suffix "dylib")
  else()
    set(link_library_suffix "so")
  endif()
elseif(WIN32)
  set(link_library_suffix "lib")
endif()
