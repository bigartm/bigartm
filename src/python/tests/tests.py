# Copyright 2014, Additive Regularization of Topic Models.

import artm.messages_pb2
import artm.library
from artm.library import *

#################################################################################
# ALL CODE BELOW DEFINES PROTOBUF MESSAGES NEED TO TEST THE INTERFACE FUNCTIONS

# Create master_config
master_config = messages_pb2.MasterComponentConfig()
master_config.processors_count = 2
master_config.processor_queue_max_size = 5
master_config.cache_theta = 1

perplexity_config = messages_pb2.PerplexityScoreConfig();
perplexity_config.stream_name = 'stream_0'

master_proxy_config = messages_pb2.MasterProxyConfig()
master_proxy_config.node_connect_endpoint = "tcp://localhost:5555"
master_proxy_config.config.CopyFrom(master_config)

stream_ = master_config.stream.add()
stream_.name = ('stream_0')
stream_.type = Stream_Type_Global
stream_.modulus = 3
stream_.residuals.append(9)

# Create batch
batch = messages_pb2.Batch()
batch.token.append('first')
item_ = batch.item.add()
item_.id = 2
field_ = item_.field.add()
field_.token_id.append(0)
field_.token_count.append(2)

# Create stream
stream = messages_pb2.Stream()
stream.name = ('stream_8')
stream.type = Stream_Type_Global
stream.modulus = 3
stream.residuals.append(1)

# Create regularizer_config
dirichlet_theta_config = messages_pb2.DirichletThetaConfig()
alpha = dirichlet_theta_config.alpha.add()
alpha.value.append(0.1)

dirichlet_phi_config = messages_pb2.DirichletPhiConfig()
dirichlet_phi_config.dictionary_name = 'dictionary_1'

# Create model_config
model_config = messages_pb2.ModelConfig()
model_config.stream_name = ('stream_0')
model_config.regularizer_name.append('regularizer_1')
model_config.regularizer_tau.append(1)
model_config.regularizer_name.append('regularizer_2')
model_config.regularizer_tau.append(2)
model_config.score_name.append('perplexity_score')

model_config_new = messages_pb2.ModelConfig()
model_config_new.CopyFrom(model_config)
model_config_new.inner_iterations_count = 20


dictionary_config = messages_pb2.DictionaryConfig()
dictionary_config.name = 'dictionary_1'

entry_1 = dictionary_config.entry.add()
entry_1.key_token = 'token_1'
entry_1.value = 0.4
entry_2 = dictionary_config.entry.add()
entry_2.key_token = 'token_2'
entry_2.value = 0.6

#################################################################################
# TEST SECTION

artm_library = Library()
with MasterComponent() as master_component:
  master_component.Reconfigure(master_config)
  perplexity_score = master_component.CreateScore('perplexity_score', ScoreConfig_Type_Perplexity, perplexity_config)
  master_component.CreateStream(stream)
  master_component.RemoveStream(stream)
  model = master_component.CreateModel(model_config)
  master_component.RemoveModel(model)
  model = master_component.CreateModel(model_config)

  dictionary = master_component.CreateDictionary(dictionary_config)
  regularizer = master_component.CreateRegularizer('regularizer_1', 0, dirichlet_theta_config)
  master_component.RemoveRegularizer(regularizer)
  regularizer = master_component.CreateRegularizer('regularizer_1', 0, dirichlet_theta_config)

  regularizer_phi = master_component.CreateRegularizer('regularizer_2', 1, dirichlet_phi_config)

  master_component.AddBatch(batch)
  model.Enable()
  master_component.InvokeIteration(10)
  master_component.WaitIdle()
  model.Synchronize(0.0)
  model.Disable()
  args_tm = messages_pb2.GetTopicModelArgs()
  args_tm.model_name = model.name()
  topic_model = master_component.GetTopicModel(None, args = args_tm)
  args_theta = messages_pb2.GetThetaMatrixArgs()
  args_theta.model_name = model.name()
  theta_matrix = master_component.GetThetaMatrix(None, args = args_theta)
  perplexity_score = perplexity_score.GetValue(model)

  model.Overwrite(topic_model);

  # Test all 'reconfigure' methods
  regularizer.Reconfigure(0, dirichlet_theta_config)
  model.Reconfigure(model_config_new)
  master_config_new = master_component.config();
  master_config_new.processors_count = 1
  master_config_new.processor_queue_max_size = 2
  master_component.Reconfigure(master_config_new)
  master_component.RemoveDictionary(dictionary)

with NodeController("tcp://*:5555") as node_controller:
  with MasterComponent(master_proxy_config) as master_component:
    master_component.Reconfigure(master_config)

print 'All tests have been successfully passed!'
