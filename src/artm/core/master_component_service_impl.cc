// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/master_component_service_impl.h"

#include <string>

#include "boost/thread.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/string_generator.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/utility.hpp"
#include "glog/logging.h"

#include "rpcz/application.hpp"

#include "artm/core/instance.h"
#include "artm/core/merger.h"
#include "artm/core/batch_manager.h"

namespace artm {
namespace core {

MasterComponentServiceImpl::MasterComponentServiceImpl(Instance* instance) : instance_(instance) {
}

void MasterComponentServiceImpl::UpdateModel(const ::artm::core::ModelIncrement& request,
                                       ::rpcz::reply< ::artm::core::Void> response) {
    instance_->merger_queue()->push(std::make_shared<::artm::core::ModelIncrement>(request));
  try {
    response.send(::artm::core::Void());
  } catch(...) {
    LOG(ERROR) << "Unable to send reply to UpdateModel.";
  }
}

void MasterComponentServiceImpl::RetrieveModel(const ::artm::core::String& request,
                                         ::rpcz::reply< ::artm::TopicModel> response) {
  ::artm::TopicModel topic_model;
  bool succeeded = instance_->merger()->RetrieveExternalTopicModel(request.value(), &topic_model);
  try {
    if (succeeded) {
      response.send(topic_model);
    } else {
      response.Error(0, "Model with requested ID was does not exist on server");
    }
  } catch(...) {
    LOG(ERROR) << "Unable to send reply to UpdateModel.";
  }
}

void MasterComponentServiceImpl::RequestBatches(const ::artm::core::Int& request,
                      ::rpcz::reply< ::artm::core::BatchIds> response) {
  BatchIds reply;
  for (int i = 0; i < request.value(); ++i) {
    boost::uuids::uuid uuid = instance_->batch_manager()->Next();
    if (uuid.is_nil()) {
      break;
    }

    reply.add_batch_id(boost::lexical_cast<std::string>(uuid));
  }

  try {
    response.send(reply);
  } catch(...) {
    LOG(ERROR) << "Unable to send reply to RequestBatches.";
  }
}

void MasterComponentServiceImpl::ReportBatches(const ::artm::core::BatchIds& request,
                      ::rpcz::reply< ::artm::core::Void> response) {
  for (int i = 0; i < request.batch_id_size(); ++i) {
    boost::uuids::uuid uuid = boost::uuids::string_generator()(request.batch_id(i));
    if (uuid.is_nil()) {
      LOG(ERROR) << "Unable to convert " << request.batch_id(i) << " to uuid.";
      continue;
    }

    instance_->batch_manager()->Done(uuid, ModelName());
  }

  try {
    response.send(Void());
  } catch(...) {
    LOG(ERROR) << "Unable to send reply to ReportBatches.";
  }
}

}  // namespace core
}  // namespace artm
