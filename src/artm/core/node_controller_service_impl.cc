// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/node_controller_service_impl.h"

#include <string>

#include "boost/thread.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"
#include "artm/core/instance.h"
#include "artm/core/master_component.h"
#include "artm/core/merger.h"

namespace artm {
namespace core {

NodeControllerServiceImpl::NodeControllerServiceImpl()
    : lock_(), instance_(nullptr) { }

NodeControllerServiceImpl::~NodeControllerServiceImpl() {}

void NodeControllerServiceImpl::CreateOrReconfigureInstance(
    const ::artm::MasterComponentConfig& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  try {
    boost::lock_guard<boost::mutex> guard(lock_);
    if (instance_ != nullptr) {
        LOG(INFO) << "Reconfigure an existing instance";
        instance_->Reconfigure(request);
    } else {
      LOG(INFO) << "Create a new instance";
      instance_.reset(new Instance(request, NodeControllerInstance));
    }

    response.send(Void());
  } CATCH_EXCEPTIONS_AND_SEND_ERROR;
}

void NodeControllerServiceImpl::DisposeInstance(
    const ::artm::core::Void& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  boost::lock_guard<boost::mutex> guard(lock_);
  if (instance_ != nullptr) {
    LOG(INFO) << "Dispose the instance";
    instance_.reset();
  }

  response.send(Void());
}

void NodeControllerServiceImpl::CreateOrReconfigureModel(
    const ::artm::core::CreateOrReconfigureModelArgs& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  try {
    boost::lock_guard<boost::mutex> guard(lock_);
    VerifyCurrentState();

    if (instance_ != nullptr) {
      instance_->CreateOrReconfigureModel(request.config());
    }

    response.send(Void());
  } CATCH_EXCEPTIONS_AND_SEND_ERROR;
}

void NodeControllerServiceImpl::DisposeModel(
    const ::artm::core::DisposeModelArgs& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  boost::lock_guard<boost::mutex> guard(lock_);
  if (instance_ != nullptr) {
    instance_->DisposeModel(request.model_name());
  }

  response.send(Void());
}

void NodeControllerServiceImpl::CreateOrReconfigureRegularizer(
    const ::artm::core::CreateOrReconfigureRegularizerArgs& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  try {
    boost::lock_guard<boost::mutex> guard(lock_);
    VerifyCurrentState();

    if (instance_ != nullptr) {
      instance_->CreateOrReconfigureRegularizer(request.config());
    }

    response.send(Void());
  } CATCH_EXCEPTIONS_AND_SEND_ERROR;
}

void NodeControllerServiceImpl::DisposeRegularizer(
    const ::artm::core::DisposeRegularizerArgs& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  boost::lock_guard<boost::mutex> guard(lock_);
  if (instance_ != nullptr) {
    instance_->DisposeRegularizer(request.regularizer_name());
  }

  response.send(Void());
}

void NodeControllerServiceImpl::CreateOrReconfigureDictionary(
    const ::artm::core::CreateOrReconfigureDictionaryArgs& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  try {
    boost::lock_guard<boost::mutex> guard(lock_);
    VerifyCurrentState();

    if (instance_ != nullptr) {
      instance_->CreateOrReconfigureDictionary(request.dictionary());
    }

    response.send(Void());
  } CATCH_EXCEPTIONS_AND_SEND_ERROR;
}

void NodeControllerServiceImpl::DisposeDictionary(
    const ::artm::core::DisposeDictionaryArgs& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  try {
    boost::lock_guard<boost::mutex> guard(lock_);
    if (instance_ != nullptr) {
      instance_->DisposeDictionary(request.dictionary_name());
    }

    response.send(Void());
  } CATCH_EXCEPTIONS_AND_SEND_ERROR;
}

void NodeControllerServiceImpl::ForcePullTopicModel(
    const ::artm::core::Void& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  try {
    boost::lock_guard<boost::mutex> guard(lock_);
    if (instance_ != nullptr) {
      instance_->merger()->ForcePullTopicModel();
    } else {
      LOG(ERROR) << "No instances exist in node controller";
    }

    response.send(Void());
  } CATCH_EXCEPTIONS_AND_SEND_ERROR;
}

void NodeControllerServiceImpl::ForcePushTopicModelIncrement(
    const ::artm::core::Void& request,
    ::rpcz::reply< ::artm::core::Void> response) {
  try {
    boost::lock_guard<boost::mutex> guard(lock_);
    if (instance_ != nullptr) {
      instance_->merger()->ForcePushTopicModelIncrement();
    } else {
      LOG(ERROR) << "No instances exist in node controller";
    }

    response.send(Void());
  } CATCH_EXCEPTIONS_AND_SEND_ERROR;
}

Instance* NodeControllerServiceImpl::instance() {
  return instance_.get();
}

void NodeControllerServiceImpl::VerifyCurrentState() {
  if (instance_ == nullptr) {
    std::string message = "Instance does not exist";
    LOG(ERROR) << message;
    BOOST_THROW_EXCEPTION(InvalidOperation(message));
  }
}

}  // namespace core
}  // namespace artm
