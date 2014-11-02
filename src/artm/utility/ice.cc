/**
 * Ice multiplatform dynamic library loader.
 *
 * @copyright 2011 David Rebbe
 */

#include "artm/utility/ice.h"

namespace ice {

Library::Library(std::string name) {
#if (defined(_WIN32) || defined(__WIN32__))
#ifdef UNICODE
  int len = MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, NULL, 0);
  if (len) {
      wchar_t* n = new wchar_t[len];
      MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, n, len);
  m_lib = LoadLibrary(n);
      delete[] n;
  } else {
    m_lib = NULL;
  }
#else
  m_lib = LoadLibrary(name.c_str());
#endif  // UNICODE
  if (m_lib == NULL) {
    DWORD error = GetLastError();
    std::stringstream err;
    err << "Failed to open library: '" << name <<
      "' with error code: #" << error;
    throw Exception(err.str());
}
#else
  m_lib = dlopen(name.c_str(), RTLD_NOW);
  if (m_lib == NULL) {
    std::stringstream err;
    err << "Failed to open library '" << name <<
      "': " << dlerror();
    throw Exception(err.str());
  }
#endif
}

Library::~Library() {
  if (this->isLoaded()) {
#if (defined(_WIN32) || defined(__WIN32__))
    FreeLibrary(m_lib);
#else
    dlclose(m_lib);
#endif
  }
}

bool Library::isLoaded() const throw() {
  return m_lib != NULL;
}

HMODULE const& Library::_library() const throw() {
  return m_lib;
}

}  // namespace ice
