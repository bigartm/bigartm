// Copyright 2017, Additive Regularization of Topic Models.

/*****************************************************************
// All exceptions in artm::core should be inherited from std::runtime_error,
// using DEFINE_EXCEPTION_TYPE (see below). Example of how to throw and handle an exception:
try {
  BOOST_THROW_EXCEPTION(InvalidOperation("This operation is invalid in current state."));
} catch (const std::runtime_error& e) {
  std::cerr << e.what() << std::endl;
  std::cerr << *boost::get_error_info<boost::throw_file>(e) << std::endl; 
  std::cerr << *boost::get_error_info<boost::throw_line>(e) << std::endl; 
  std::cerr << *boost::get_error_info<boost::throw_function>(e) << std::endl; 
}
*****************************************************************/

#ifndef SRC_ARTM_CORE_EXCEPTIONS_H_
#define SRC_ARTM_CORE_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

#include "boost/exception/diagnostic_information.hpp"
#include "boost/exception/get_error_info.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/throw_exception.hpp"

#ifndef ARTM_ERROR_CODES_EXIST
#define ARTM_ERROR_CODES_EXIST
enum ArtmErrorCodes {
    ARTM_SUCCESS = 0,                   // Has no corresponding exception type.
    ARTM_STILL_WORKING = -1,            // Has no corresponding exception type.
    ARTM_INTERNAL_ERROR = -2,
    ARTM_ARGUMENT_OUT_OF_RANGE = -3,
    ARTM_INVALID_MASTER_ID = -4,
    ARTM_CORRUPTED_MESSAGE = -5,
    ARTM_INVALID_OPERATION = -6,
    ARTM_DISK_READ_ERROR = -7,
    ARTM_DISK_WRITE_ERROR = -8,
};
#endif

namespace artm {
namespace core {

#define DEFINE_EXCEPTION_TYPE(Type, BaseType)          \
class Type : public BaseType { public:  /*NOLINT*/     \
  explicit Type(std::string what) : BaseType(what) { }  \
  explicit Type(const char* what) : BaseType(what) { }  \
};

DEFINE_EXCEPTION_TYPE(InternalError, std::runtime_error);
class ArgumentOutOfRangeException : public std::runtime_error {
 public:
  template<class T>
  explicit ArgumentOutOfRangeException(std::string argument, T actual)
      : std::runtime_error(argument + " == " +
        boost::lexical_cast<std::string>(actual) + ", out of range.") { }
  template<class T>
  explicit ArgumentOutOfRangeException(std::string argument, T actual, std::string message)
      : std::runtime_error(argument + " == " +
        boost::lexical_cast<std::string>(actual) + ", out of range. " + message) { }
};
DEFINE_EXCEPTION_TYPE(InvalidMasterIdException, std::runtime_error);
DEFINE_EXCEPTION_TYPE(CorruptedMessageException, std::runtime_error);
DEFINE_EXCEPTION_TYPE(InvalidOperation, std::runtime_error);
DEFINE_EXCEPTION_TYPE(DiskReadException, std::runtime_error);
DEFINE_EXCEPTION_TYPE(DiskWriteException, std::runtime_error);

#undef DEFINE_EXCEPTION_TYPE

#define CATCH_EXCEPTIONS                                                       \
catch (const ::artm::core::InternalError& e) {                                 \
  set_last_error("InternalError :  " + std::string(e.what()));                 \
  return ARTM_INTERNAL_ERROR;                                                  \
} catch (const ::artm::core::ArgumentOutOfRangeException& e) {                 \
  set_last_error("ArgumentOutOfRangeException :  " + std::string(e.what()));   \
  return ARTM_ARGUMENT_OUT_OF_RANGE;                                           \
} catch (const ::artm::core::InvalidMasterIdException& e) {                    \
  set_last_error("InvalidMasterIdException :  " + std::string(e.what()));      \
  return ARTM_INVALID_MASTER_ID;                                               \
} catch (const ::artm::core::CorruptedMessageException& e) {                   \
  set_last_error("CorruptedMessageException :  " + std::string(e.what()));     \
  return ARTM_CORRUPTED_MESSAGE;                                               \
} catch (const ::artm::core::InvalidOperation& e) {                            \
  set_last_error("InvalidOperation :  " + std::string(e.what()));              \
  return ARTM_INVALID_OPERATION;                                               \
} catch (const ::artm::core::DiskReadException& e) {                           \
  set_last_error("DiskReadException :  " + std::string(e.what()));             \
  return ARTM_DISK_READ_ERROR;                                                 \
} catch (const ::artm::core::DiskWriteException& e) {                          \
  set_last_error("DiskWriteException :  " + std::string(e.what()));            \
  return ARTM_DISK_WRITE_ERROR;                                                \
} catch (...) {                                                                \
  LOG(ERROR) << boost::current_exception_diagnostic_information();             \
  set_last_error(boost::current_exception_diagnostic_information());           \
  return ARTM_INTERNAL_ERROR;                                                  \
}

#define CATCH_EXCEPTIONS_AND_SEND_ERROR                                        \
catch (const InternalError& e) {                                               \
  response.Error(ARTM_INTERNAL_ERROR, e.what());                               \
} catch (const ArgumentOutOfRangeException& e) {                               \
  response.Error(ARTM_ARGUMENT_OUT_OF_RANGE, e.what());                        \
} catch (const InvalidMasterIdException& e) {                                  \
  response.Error(ARTM_INVALID_MASTER_ID, e.what());                            \
} catch (const CorruptedMessageException& e) {                                 \
  response.Error(ARTM_CORRUPTED_MESSAGE, e.what());                            \
} catch (const InvalidOperation& e) {                                          \
  response.Error(ARTM_INVALID_OPERATION, e.what());                            \
} catch (const DiskReadException& e) {                                         \
  response.Error(ARTM_DISK_READ_ERROR, e.what());                              \
} catch (const DiskWriteException& e) {                                        \
  response.Error(ARTM_DISK_WRITE_ERROR, e.what());                             \
} catch (const std::runtime_error& e) {                                        \
  response.Error(ARTM_INTERNAL_ERROR, e.what());                               \
} catch (...) {                                                                \
  LOG(ERROR) << "unknown critical error.";                                     \
  response.Error(ARTM_INTERNAL_ERROR);                                         \
}

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_EXCEPTIONS_H_
