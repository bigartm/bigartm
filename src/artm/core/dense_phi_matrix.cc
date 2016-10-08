// Copyright 2015, Additive Regularization of Topic Models.

#include "artm/core/dense_phi_matrix.h"

#include <algorithm>

#include "artm/core/helpers.h"

namespace artm {
namespace core {

// =======================================================
// TokenCollection methods
// =======================================================

int TokenCollection::AddToken(const Token& token) {
  int token_id = this->token_id(token);
  if (token_id != -1)
    return token_id;

  token_id = token_size();
  token_to_token_id_.insert(
    std::make_pair(token, token_id));
  token_id_to_token_.push_back(token);
  return token_id;
}

void TokenCollection::Swap(TokenCollection* rhs) {
  token_to_token_id_.swap(rhs->token_to_token_id_);
  token_id_to_token_.swap(rhs->token_id_to_token_);
}

bool TokenCollection::has_token(const Token& token) const {
  return token_to_token_id_.count(token) > 0;
}

int TokenCollection::token_id(const Token& token) const {
  auto iter = token_to_token_id_.find(token);
  return (iter != token_to_token_id_.end()) ? iter->second : -1;
}

const Token& TokenCollection::token(int index) const {
  return token_id_to_token_[index];
}

void TokenCollection::Clear() {
  token_to_token_id_.clear();
  token_id_to_token_.clear();
}

int TokenCollection::token_size() const {
  return static_cast<int>(token_to_token_id_.size());
}

// =======================================================
// SpinLock methods
// =======================================================

void SpinLock::Lock() {
  while (state_.exchange(kLocked, std::memory_order_acquire) == kLocked) {
    /* busy-wait */
  }
}

void SpinLock::Unlock() {
  state_.store(kUnlocked, std::memory_order_release);
}

// =======================================================
// PhiMatrixFrame methods
// =======================================================

PhiMatrixFrame::PhiMatrixFrame(const ModelName& model_name,
                               const google::protobuf::RepeatedPtrField<std::string>& topic_name)
    : model_name_(model_name), topic_name_(), token_collection_(), spin_locks_() {
  for (auto iter = topic_name.begin(); iter != topic_name.end(); ++iter) {
    topic_name_.push_back(*iter);
  }
}

PhiMatrixFrame::PhiMatrixFrame(const PhiMatrixFrame& rhs)
    : model_name_(rhs.model_name_),
      topic_name_(rhs.topic_name_),
      token_collection_(rhs.token_collection_),
      spin_locks_() {
  spin_locks_.reserve(rhs.spin_locks_.size());
  for (unsigned i = 0; i < rhs.spin_locks_.size(); ++i)
    spin_locks_.push_back(std::make_shared<SpinLock>());
}

const Token& PhiMatrixFrame::token(int index) const {
  return token_collection_.token(index);
}

google::protobuf::RepeatedPtrField<std::string> PhiMatrixFrame::topic_name() const {
  google::protobuf::RepeatedPtrField<std::string> topic_name;
  for (const auto& elem : topic_name_) {
    std::string* name = topic_name.Add();
    *name = elem;
  }
  return topic_name;
}

const std::string& PhiMatrixFrame::topic_name(int topic_id) const {
  return topic_name_[topic_id];
}

std::string PhiMatrixFrame::model_name() const {
  return model_name_;
}

bool PhiMatrixFrame::has_token(const Token& token) const {
  return token_collection_.has_token(token);
}

int PhiMatrixFrame::token_index(const Token& token) const {
  return token_collection_.token_id(token);
}

void PhiMatrixFrame::Clear() {
  token_collection_.Clear();
  spin_locks_.clear();
}

int PhiMatrixFrame::AddToken(const Token& token) {
  int token_id = token_collection_.token_id(token);
  if (token_id != -1)
    return token_id;

  spin_locks_.push_back(std::make_shared<SpinLock>());
  return token_collection_.AddToken(token);
}

void PhiMatrixFrame::Swap(PhiMatrixFrame* rhs) {
  model_name_.swap(rhs->model_name_);
  topic_name_.swap(rhs->topic_name_);
  token_collection_.Swap(&rhs->token_collection_);
  spin_locks_.swap(rhs->spin_locks_);
}

// =======================================================
// PackedValues methods
// =======================================================

PackedValues::PackedValues() {}

PackedValues::PackedValues(int size) {
  values_.resize(size, 0.0f);
}

PackedValues::PackedValues(const PackedValues& rhs)
    : values_(rhs.values_), bitmask_(rhs.bitmask_), ptr_(rhs.ptr_) {}

PackedValues::PackedValues(const float* values, int size) {
  values_.resize(size); memcpy(&values_[0], values, sizeof(float) * size);
}

bool PackedValues::is_packed() const {
  return false;
}

float PackedValues::get(int index) const {
  return values_[index];
}

float* PackedValues::unpack() {
  return &values_[0];
}

void PackedValues::pack() {
}

void PackedValues::reset(int size) {
  values_.resize(size, 0.0f);
}


// =======================================================
// DensePhiMatrix methods
// =======================================================

DensePhiMatrix::DensePhiMatrix(const ModelName& model_name,
                               const google::protobuf::RepeatedPtrField<std::string>& topic_name)
    : PhiMatrixFrame(model_name, topic_name), values_() {}

DensePhiMatrix::DensePhiMatrix(const DensePhiMatrix& rhs) : PhiMatrixFrame(rhs), values_() {
  for (int token_index = 0; token_index < rhs.token_size(); ++token_index) {
    values_.push_back(PackedValues(rhs.values_[token_index]));
  }
}

DensePhiMatrix::DensePhiMatrix(const AttachedPhiMatrix& rhs)
    : PhiMatrixFrame(rhs), values_() {
  for (int token_index = 0; token_index < rhs.token_size(); ++token_index) {
    values_.push_back(PackedValues(rhs.values_[token_index], rhs.topic_size()));
  }
}

std::shared_ptr<PhiMatrix> DensePhiMatrix::Duplicate() const {
  return std::shared_ptr<PhiMatrix>(new DensePhiMatrix(*this));
}

float DensePhiMatrix::get(int token_id, int topic_id) const {
  return values_[token_id].get(topic_id);
}

void DensePhiMatrix::set(int token_id, int topic_id, float value) {
  values_[token_id].unpack()[topic_id] = value;
  if ((topic_id + 1) == topic_size())
    values_[token_id].pack();
}

void DensePhiMatrix::increase(int token_id, int topic_id, float increment) {
  values_[token_id].unpack()[topic_id] += increment;
  if ((topic_id + 1) == topic_size())
    values_[token_id].pack();
}

void DensePhiMatrix::increase(int token_id, const std::vector<float>& increment) {
  const int topic_size = this->topic_size();
  assert(increment.size() == topic_size);
  float* values = values_[token_id].unpack();

  this->Lock(token_id);
  for (int topic_index = 0; topic_index < topic_size; ++topic_index)
    values[topic_index] += increment[topic_index];
  this->Unlock(token_id);
}

void DensePhiMatrix::Clear() {
  values_.clear();
  PhiMatrixFrame::Clear();
}

int DensePhiMatrix::AddToken(const Token& token) {
  int token_id = token_index(token);
  if (token_id != -1)
    return token_id;

  values_.push_back(PackedValues(topic_size()));
  int retval = PhiMatrixFrame::AddToken(token);
  assert(retval == (values_.size() - 1));
  return retval;
}

void DensePhiMatrix::Reset() {
  for (PackedValues& value : values_)
    value.reset(topic_size());
}

void DensePhiMatrix::Reshape(const PhiMatrix& phi_matrix) {
  Clear();
  for (int token_id = 0; token_id < phi_matrix.token_size(); ++token_id) {
    this->AddToken(phi_matrix.token(token_id));
  }
}

// =======================================================
// AttachedPhiMatrix methods
// =======================================================

AttachedPhiMatrix::AttachedPhiMatrix(int address_length, float* address, PhiMatrixFrame* source)
    : PhiMatrixFrame(source->model_name(), source->topic_name()) {

  int topic_size = source->topic_size();
  int token_size = source->token_size();

  if (topic_size * token_size * sizeof(float) != address_length) {
    std::stringstream ss;
    ss << "Pointer " << address_length << " (" << address_length << "bytes) is incompatible with model "
       << source->model_name() << " (|T|=" << topic_size << ", |W|=" << token_size << ")";
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(ss.str()));
  }

  for (int token_index = 0; token_index < token_size; ++token_index) {
    float* token_address = address + topic_size * token_index;
    values_.push_back(token_address);
    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      token_address[topic_index] = source->get(token_index, topic_index);
    }
  }

  PhiMatrixFrame::Swap(source);
  source->Clear();
}

std::shared_ptr<PhiMatrix> AttachedPhiMatrix::Duplicate() const {
  return std::shared_ptr<PhiMatrix>(new DensePhiMatrix(*this));
}

void AttachedPhiMatrix::increase(int token_id, const std::vector<float>& increment) {
  const int topic_size = this->topic_size();
  assert(increment.size() == topic_size);
  float* values = values_[token_id];

  this->Lock(token_id);
  for (int topic_index = 0; topic_index < topic_size; ++topic_index)
    values[topic_index] += increment[topic_index];
  this->Unlock(token_id);
}

void AttachedPhiMatrix::Clear() {
  values_.clear();
  PhiMatrixFrame::Clear();
}

int AttachedPhiMatrix::AddToken(const Token& token) {
  BOOST_THROW_EXCEPTION(artm::core::InternalError("Tokens addition is not allowed for attached model."));
}

}  // namespace core
}  // namespace artm
