// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/cache_manager.h"

#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/core/helpers.h"
#include "artm/core/instance.h"
#include "artm/core/dense_phi_matrix.h"
#include "artm/core/protobuf_helpers.h"

namespace fs = boost::filesystem;

namespace artm {
namespace core {

ThetaCacheEntry::ThetaCacheEntry()
    : theta_matrix_(std::make_shared<ThetaMatrix>())
    , filename_() { }

ThetaCacheEntry::~ThetaCacheEntry() {
  if (!filename_.empty()) {
    try { fs::remove(fs::path(filename_)); }
    catch (...) { }
  }
}

CacheManager::CacheManager(const std::string& disk_path, Instance* instance)
    : lock_()
    , disk_path_(disk_path)
    , instance_(instance)
    , cache_() {
  Clear();
}

CacheManager::~CacheManager() {
  cache_.clear();
}

void CacheManager::Clear() {
  cache_.clear();
  std::string ptd_name = (instance_ != nullptr) ? instance_->config()->ptd_name() : std::string();
  if (!ptd_name.empty()) {
    std::shared_ptr<PhiMatrix> ptd(new DensePhiMatrix(ptd_name, instance_->config()->topic_name()));
    instance_->SetPhiMatrix(ptd_name, ptd);
  }
}

void CacheManager::RequestMasterComponentInfo(MasterComponentInfo* master_info) const {
  for (const auto& key : cache_.keys()) {
    std::shared_ptr<ThetaCacheEntry> entry = cache_.get(key);
    if (entry == nullptr) {
      continue;
    }

    MasterComponentInfo::CacheEntryInfo* info = master_info->add_cache_entry();
    info->set_key(boost::lexical_cast<std::string>(key));
    info->set_byte_size(entry->theta_matrix()->ByteSize());
  }
}

// ToDo(sashafrey): this method has grown too big and complicated.
// It needs to be refactored.
static bool PopulateThetaMatrixFromCacheEntry(const ThetaMatrix& cache,
                                              const GetThetaMatrixArgs& get_theta_args,
                                              ::artm::ThetaMatrix* theta_matrix) {
  auto& args_topic_name = get_theta_args.topic_name();
  const bool has_sparse_format = get_theta_args.matrix_layout() == MatrixLayout_Sparse;
  const bool sparse_cache = cache.topic_indices_size() > 0;
  bool use_all_topics = false;

  std::vector<int> topics_to_use;
  if (args_topic_name.size() != 0) {
    for (int i = 0; i < args_topic_name.size(); ++i) {
      int topic_index = repeated_field_index_of(cache.topic_name(), args_topic_name.Get(i));
      if (topic_index == -1) {
        std::stringstream ss;
        ss << "GetThetaMatrixArgs.topic_name[" << i << "] == " << args_topic_name.Get(i)
           << " does not exist in MasterModelConfig.topic_name";
        BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(ss.str()));
      }

      assert(topic_index >= 0 && topic_index < cache.topic_name_size());
      topics_to_use.push_back(topic_index);
    }
  } else {  // use all topics
    assert(cache.topic_name_size() > 0);
    for (int i = 0; i < cache.topic_name_size(); ++i) {
      topics_to_use.push_back(i);
    }
    use_all_topics = true;
  }

  // Populate num_topics and topic_name fields in the resulting message
  ::google::protobuf::RepeatedPtrField< ::std::string> result_topic_name;
  for (int topic_index : topics_to_use) {
    result_topic_name.Add()->assign(cache.topic_name(topic_index));
  }

  if (theta_matrix->topic_name_size() == 0) {
    // Assign
    theta_matrix->set_num_topics(result_topic_name.size());
    assert(theta_matrix->topic_name_size() == 0);
    for (const TopicName& topic_name : result_topic_name) {
      theta_matrix->add_topic_name(topic_name);
    }
  } else {
    // Verify
    if (theta_matrix->num_topics() != result_topic_name.size()) {
      BOOST_THROW_EXCEPTION(artm::core::InternalError("theta_matrix->num_topics() != result_topic_name.size()"));
    }

    for (int i = 0; i < theta_matrix->topic_name_size(); ++i) {
      if (theta_matrix->topic_name(i) != result_topic_name.Get(i)) {
        BOOST_THROW_EXCEPTION(artm::core::InternalError("theta_matrix->topic_name(i) != result_topic_name.Get(i)"));
      }
    }
  }

  bool has_title = (cache.item_title_size() == cache.item_id_size());
  for (int item_index = 0; item_index < cache.item_id_size(); ++item_index) {
    theta_matrix->add_item_id(cache.item_id(item_index));
    if (has_title) {
      theta_matrix->add_item_title(cache.item_title(item_index));
    }
    ::artm::FloatArray* theta_vec = theta_matrix->add_item_weights();

    const artm::FloatArray& item_theta = cache.item_weights(item_index);
    if (!has_sparse_format) {
      if (sparse_cache) {
        // dense output -- sparse cache
        for (unsigned index = 0; index < topics_to_use.size(); ++index) {
          int topic_index = repeated_field_index_of(cache.topic_indices(item_index).value(), topics_to_use[index]);
          theta_vec->add_value(topic_index != -1 ? item_theta.value(topic_index) : 0.0f);
        }
      } else {
        // dense output -- dense cache
        for (int topic_index : topics_to_use) {
          theta_vec->add_value(item_theta.value(topic_index));
        }
      }
    } else {
      ::artm::IntArray* sparse_topic_indices = theta_matrix->add_topic_indices();
      if (sparse_cache) {
        // sparse output -- sparse cache
        for (int index = 0; index < cache.topic_indices(item_index).value_size(); ++index) {
          int topic_index = cache.topic_indices(item_index).value(index);
          if (use_all_topics) {
            theta_vec->add_value(item_theta.value(index));
            sparse_topic_indices->add_value(topic_index);
          } else {
            for (unsigned i = 0; i < topics_to_use.size(); ++i) {
              if (topics_to_use[i] == topic_index) {
                theta_vec->add_value(item_theta.value(index));
                sparse_topic_indices->add_value(topic_index);
                break;
              }
            }
          }
        }
      } else {
        // sparse output -- dense cache
        for (unsigned index = 0; index < topics_to_use.size(); index++) {
          int topic_index = topics_to_use[index];
          float value = item_theta.value(topic_index);
          if (value >= get_theta_args.eps()) {
            theta_vec->add_value(value);
            sparse_topic_indices->add_value(index);
          }
        }
      }
    }
  }

  return true;
}

void CacheManager::RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                                      ::artm::ThetaMatrix* theta_matrix) const {
  std::string ptd_name = (instance_ != nullptr) ? instance_->config()->ptd_name() : std::string();
  if (!ptd_name.empty()) {
    boost::lock_guard<boost::mutex> guard(lock_);
    std::shared_ptr<const ::artm::core::PhiMatrix> phi_matrix = instance_->GetPhiMatrixSafe(ptd_name);
    ThetaMatrix cached_theta;
    cached_theta.mutable_topic_name()->CopyFrom(phi_matrix->topic_name());
    std::vector<float> values; values.resize(phi_matrix->topic_size());
    for (int token_id = 0; token_id < phi_matrix->token_size(); token_id++) {
      Token token = phi_matrix->token(token_id);
      cached_theta.add_item_title(token.keyword);
      cached_theta.add_item_id(-1);  // not available
      ::artm::FloatArray* item_weights = cached_theta.add_item_weights();
      phi_matrix->get(token_id, &values);
      for (int topic_index = 0; topic_index < phi_matrix->topic_size(); topic_index++) {
        item_weights->add_value(values[topic_index]);
      }
    }

    PopulateThetaMatrixFromCacheEntry(cached_theta, get_theta_args, theta_matrix);
    return;
  }

  auto keys = cache_.keys();
  for (const auto &key : keys) {
    std::shared_ptr<ThetaMatrix> cached_theta = FindCacheEntry(key);
    if (cached_theta != nullptr) {
      PopulateThetaMatrixFromCacheEntry(*cached_theta, get_theta_args, theta_matrix);
    }
  }
}

std::shared_ptr<ThetaMatrix> CacheManager::FindCacheEntry(const Batch& batch) const {
  std::string ptd_name = (instance_ != nullptr) ? instance_->config()->ptd_name() : std::string();
  if (!ptd_name.empty()) {
    boost::lock_guard<boost::mutex> guard(lock_);
    std::shared_ptr<const ::artm::core::PhiMatrix> phi_matrix = instance_->GetPhiMatrixSafe(ptd_name);
    auto cached_theta = std::make_shared<ThetaMatrix>();
    cached_theta->mutable_topic_name()->CopyFrom(phi_matrix->topic_name());
    std::vector<float> values; values.resize(phi_matrix->topic_size());
    for (int item_id = 0; item_id < batch.item_size(); item_id++) {
      Token token(DocumentsClass, batch.item(item_id).title());

      if (token.keyword.empty()) {
        continue;
      }

      int token_index = phi_matrix->token_index(token);
      if (token_index < 0) {
        continue;
      }

      cached_theta->add_item_title(batch.item(item_id).title());
      cached_theta->add_item_id(batch.item(item_id).id());
      ::artm::FloatArray* item_weights = cached_theta->add_item_weights();
      phi_matrix->get(token_index, &values);
      for (int topic_index = 0; topic_index < phi_matrix->topic_size(); topic_index++) {
        item_weights->add_value(values[topic_index]);
      }
    }

    return cached_theta;
  }

  return FindCacheEntry(batch.id());
}

std::shared_ptr<ThetaMatrix> CacheManager::FindCacheEntry(const std::string& batch_id) const {
  std::shared_ptr<ThetaCacheEntry> retval = cache_.get(batch_id);
  if (retval == nullptr) {
    return nullptr;
  }
  if (retval->filename().empty()) {
    return retval->theta_matrix();
  }

  try {
    std::shared_ptr<ThetaMatrix> copy(std::make_shared<ThetaMatrix>());
    Helpers::LoadMessage(retval->filename(), copy.get());
    return copy;
  } catch (...) {
    LOG(ERROR) << "Unable to reload cache for " << retval->filename();
  }

  return nullptr;
}

void CacheManager::UpdateCacheEntry(const std::string& batch_id, const ThetaMatrix& theta_matrix) const {
  std::string ptd_name = (instance_ != nullptr) ? instance_->config()->ptd_name() : std::string();
  if (!ptd_name.empty()) {
    boost::lock_guard<boost::mutex> guard(lock_);
    std::shared_ptr<const ::artm::core::PhiMatrix> phi_matrix = instance_->GetPhiMatrixSafe(ptd_name);
    PhiMatrix* mutable_phi_matrix = const_cast<PhiMatrix*>(phi_matrix.get());
    for (int i = 0; i < theta_matrix.item_title_size(); i++) {
      Token token(DocumentsClass, theta_matrix.item_title(i));
      int token_id = phi_matrix->token_index(token);
      if (token_id < 0) {
        token_id = mutable_phi_matrix->AddToken(token);
      }
      for (int topic_index = 0; topic_index < theta_matrix.topic_name_size(); topic_index++) {
        mutable_phi_matrix->set(token_id, topic_index, theta_matrix.item_weights(i).value(topic_index));
      }
    }
    return;
  }

  std::shared_ptr<ThetaCacheEntry> new_entry(std::make_shared<ThetaCacheEntry>());
  if (disk_path_.empty()) {
    new_entry->theta_matrix()->CopyFrom(theta_matrix);
  } else {
    boost::uuids::uuid uuid = boost::uuids::random_generator()();
    fs::path file(boost::lexical_cast<std::string>(uuid) + ".cache");
    try {
      Helpers::SaveMessage(file.string(), disk_path_, theta_matrix);
      new_entry->mutable_filename()->assign((fs::path(disk_path_) / file).string());
    } catch (...) {
      LOG(ERROR) << "Unable to save cache entry to " << disk_path_;
      cache_.set(batch_id, nullptr);
      return;
    }
  }

  cache_.set(batch_id, new_entry);
}

void CacheManager::CopyFrom(const CacheManager& cache_manager) {
  disk_path_ = cache_manager.disk_path_;
  auto keys = cache_manager.cache_.keys();
  for (const auto& key : keys) {
    cache_.set(key, cache_manager.cache_.get(key));
  }
}

}  // namespace core
}  // namespace artm
