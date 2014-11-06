/**
 * Ice multiplatform dynamic library loader.
 *
 * @copyright 2011 David Rebbe
 */

#ifndef SRC_ARTM_UTILITY_ICE_H_
#define SRC_ARTM_UTILITY_ICE_H_

#if (defined(_WIN32) || defined(__WIN32__))
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #define VC_EXTRALEAN
  #endif  // WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <dlfcn.h>
  #define HMODULE void*
#endif

#include <stdexcept>
#include <string>
#include <sstream>

namespace ice {
  class Library;
  class Exception;
  template <class Signature> class Function;
};

class ice::Library {
 private:
  HMODULE m_lib;
  std::string m_name;

 public:
  explicit Library(std::string name);
  ~Library();
  bool isLoaded() const throw();
  std::string name() const { return m_name; }

  HMODULE const& _library() const throw();
};

class ice::Exception : std::exception {
 private:
  const std::string m_msg;

 public:
  explicit Exception(const std::string msg) : m_msg(msg) {}
  ~Exception() throw() {}
  const char* what() const throw() { return this->m_msg.c_str(); }
  std::string const whatString() const throw() { return this->m_msg; }
};

template <class Signature>
class ice::Function {
 private:
  ice::Library* m_lib;

 public:
  Function(ice::Library* library, std::string name) : m_name(name) {
    m_lib = library;
    if (library == NULL) {
      std::stringstream ss;
      ss << "Library is NULL, can't call function: '" <<
        name << "'";
      throw ice::Exception(ss.str());
    }
#if (defined(_WIN32) || defined(__WIN32__))
      m_func = reinterpret_cast<Signature*>(GetProcAddress(library->_library(), name.c_str()));
      if (m_func == NULL) {
        std::stringstream err;
        err << "Failed to Retrieve address of function '" << name <<
          "': Windows Error #" << GetLastError();
        throw ice::Exception(err.str());
      }
#else
      m_func = reinterpret_cast<Signature*>(dlsym(library->_library(), name.c_str()));
      if (m_func == NULL) {
        std::stringstream err;
        err << "Failed to Retrieve address of function '" << name <<
          "': " << dlerror();
        throw ice::Exception(err.str());
      }
#endif
  }
  operator Signature *() const throw(ice::Exception) {
    if (m_func == NULL) {
      std::stringstream ss;
      ss << "Function address '" << m_name + "' isn't resolved for library: '" <<
        m_lib->name() << "'";
      throw ice::Exception(ss.str());
    }
    return m_func;
  }
  bool isValid() const throw() { return m_func != NULL; }
  std::string name() const throw() { return m_name; }

 protected:
  Signature* m_func;
  const std::string m_name;
};

#endif  // SRC_ARTM_UTILITY_ICE_H_
