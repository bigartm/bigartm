// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_CALL_ON_DESTRUCTION_H_
#define SRC_ARTM_CORE_CALL_ON_DESTRUCTION_H_

#include <functional>

namespace artm {
namespace core {

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&);             \
    void operator=(const TypeName&)        \

// An object that accepts a lambda expression end executes it in destructor
class call_on_destruction {
 public:
  call_on_destruction(std::function<void()> f) : f_(f) {}
  ~call_on_destruction() { f_(); }

 private:
  std::function<void()> f_;
  DISALLOW_COPY_AND_ASSIGN(call_on_destruction);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_CALL_ON_DESTRUCTION_H_
