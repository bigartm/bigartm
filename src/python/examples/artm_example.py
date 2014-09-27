import artm.messages_pb2
import artm.library
from artm.library import *

import sys, glob, random

artm_library = Library()

# Parse collection
batches_found = len(glob.glob("kos/*.batch"))
if batches_found == 0:
  print "No batches found, parsing them from textual collection...",
  collection_parser_config = messages_pb2.CollectionParserConfig();
  collection_parser_config.format = CollectionParserConfig_Format_BagOfWordsUci
  
  data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ""
  collection_parser_config.docword_file_path = data_folder + 'docword.kos.txt'
  collection_parser_config.vocab_file_path = data_folder + 'vocab.kos.txt'
  collection_parser_config.target_folder = 'kos'
  collection_parser_config.dictionary_file_name = 'dictionary'
  unique_tokens = artm_library.ParseCollection(collection_parser_config);
  print " OK."
else:
  print "Found " + str(batches_found) + " batches, using them."
  unique_tokens = artm_library.LoadDictionary('kos/dictionary');

# Create master component and infer topic model
master_component_config = messages_pb2.MasterComponentConfig()
master_component_config.disk_path = 'kos'
master_component_config.processors_count = 2
with artm.library.MasterComponent(master_component_config) as master:

  # Configure a perplexity score calculator
  master.CreateScore('perplexity_score',
    ScoreConfig_Type_Perplexity,
    messages_pb2.PerplexityScoreConfig())

  # Configure a theta sparsity score calculator
  master.CreateScore('theta_sparsity_score',
    ScoreConfig_Type_SparsityTheta,
    messages_pb2.SparsityThetaScoreConfig())

  # Configure a phi sparsity score calculator
  master.CreateScore('phi_sparsity_score',
    ScoreConfig_Type_SparsityPhi,
    messages_pb2.SparsityPhiScoreConfig())

  # Configure a top tokens score calculator
  master.CreateScore(
    'top_tokens_score',
    ScoreConfig_Type_TopTokens,
    messages_pb2.TopTokensScoreConfig())

  # Configure a theta matrix snippet score calculator
  theta_snippet_config = messages_pb2.ThetaSnippetScoreConfig()
  for i in range(1, 11): theta_snippet_config.item_id.append(i)
  master.CreateScore(
    'theta_snippet_score',
    ScoreConfig_Type_ThetaSnippet,
	theta_snippet_config)

  # Configure basic regularizers
  master.CreateRegularizer(
    'reg_theta',
    RegularizerConfig_Type_DirichletTheta,
    messages_pb2.DirichletThetaConfig())

  master.CreateRegularizer(
    'reg_phi',
    RegularizerConfig_Type_DirichletPhi,
    messages_pb2.DirichletPhiConfig())

  master.CreateRegularizer(
    'reg_decorrelator',
    RegularizerConfig_Type_DecorrelatorPhi,
    messages_pb2.DecorrelatorPhiConfig())

  # Configure the model
  model_config = messages_pb2.ModelConfig()
  model_config.topics_count = 9
  model_config.inner_iterations_count = 10
  model_config.score_name.append("perplexity_score")
  model_config.score_name.append("phi_sparsity_score")
  model_config.score_name.append("theta_sparsity_score")
  model_config.score_name.append("top_tokens_score")
  model_config.score_name.append("theta_snippet_score")
  model_config.regularizer_name.append('reg_theta')
  model_config.regularizer_tau.append(-0.1)
  model_config.regularizer_name.append('reg_phi')
  model_config.regularizer_tau.append(-0.2)
  model_config.regularizer_name.append('reg_decorrelator')
  model_config.regularizer_tau.append(1000000)
  model = master.CreateModel(model_config)

  random.seed(123)
  initial_topic_model = messages_pb2.TopicModel();
  initial_topic_model.topics_count = model_config.topics_count;
  initial_topic_model.name = model.name()
  for i in range(0, len(unique_tokens.entry)):
    token = unique_tokens.entry[i].key_token
    initial_topic_model.token.append(token);
    weights = initial_topic_model.token_weights.add();
    for topic_index in range(0, model_config.topics_count):
      weights.value.append(random.random())
  model.Overwrite(initial_topic_model)

  for iter in range(0, 8):
    master.InvokeIteration(1)        # Invoke one scan of the entire collection...
    master.WaitIdle();               # and wait until it completes.
    model.Synchronize(0.0);          # Synchronize topic model.
    perplexity_score = master.GetScore(model, 'perplexity_score')
    sparsity_phi_score = master.GetScore(model, 'phi_sparsity_score')
    sparsity_theta_score = master.GetScore(model, 'theta_sparsity_score')
    print "Iter#" + str(iter),
    print ": Perplexity = %.3f" % perplexity_score.value,
    print ", Phi sparsity = %.3f " % sparsity_phi_score.value,
    print ", Theta sparsity = %.3f" % sparsity_theta_score.value

  print '\nTop tokens per topic:'
  top_tokens_score = master.GetScore(model, 'top_tokens_score')
  for i in range(0, len(top_tokens_score.values)):
    print "Topic#" + str(i+1) + ": ",
    for value in top_tokens_score.values[i].value:
      print value + " ",
    print "\n",

  print '\nSnippet of theta matrix:'
  theta_snippet_score = master.GetScore(model, 'theta_snippet_score')
  for i in range(0, len(theta_snippet_score.values)):
    print "Item#" + str(theta_snippet_score.item_id[i]) + ": ",
    for value in theta_snippet_score.values[i].value:
      print "%.3f\t" % value,
    print "\n",
