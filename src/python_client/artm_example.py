import sys
sys.path.append('src')

from python_interface import *
import glob

os.environ['PATH'] = ';'.join([os.path.abspath(os.curdir) + '\\bin', os.environ['PATH']])
library = ArtmLibrary(os.path.abspath(os.curdir) + '\\bin\\artm.dll')

# Parse collection
batches_found = len(glob.glob("kos/*.batch"))
collection_parser_config = messages_pb2.CollectionParserConfig();
collection_parser_config.target_folder = 'kos'
collection_parser_config.dictionary_file_name = 'dictionary'
if batches_found != 0:
  collection_parser_config.format = CollectionParserConfig_Format_JustLoadDictionary
else:
  collection_parser_config.format = CollectionParserConfig_Format_BagOfWordsUci
  collection_parser_config.docword_file_path = 'docword.kos.txt'
  collection_parser_config.vocab_file_path = 'vocab.kos.txt'
library.ParseCollection(collection_parser_config);

# Create master component and infer topic model
master_component_config = messages_pb2.MasterComponentConfig()
master_component_config.disk_path = 'kos'
with library.CreateMasterComponent(master_component_config) as master:
  # Setup basic configuration
  master_config = master.config()
  master_config.processors_count = 2
  master_config.cache_theta = 1
  master.Reconfigure(master_config)

  # Configure a perplexity score calculator
  master.CreateScore(
    'perplexity_score',
    ScoreConfig_Type_Perplexity,
    messages_pb2.PerplexityScore())

  # Configure a top tokens score calculator
  master.CreateScore(
    'top_tokens_score',
    ScoreConfig_Type_TopTokens,
    messages_pb2.TopTokensScoreConfig())

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
  model_config.topics_count = 4
  model_config.inner_iterations_count = 10
  model_config.score_name.append("perplexity_score")
  model_config.score_name.append("top_tokens_score")
  model_config.regularizer_name.append('reg_theta')
  model_config.regularizer_tau.append(0.1)
  model_config.regularizer_name.append('reg_phi')
  model_config.regularizer_tau.append(-0.1)
  model_config.regularizer_name.append('reg_decorrelator')
  model_config.regularizer_tau.append(10000)
  model = master.CreateModel(model_config)

  for iter in range(1, 8):
    master.InvokeIteration(1)        # Invoke one scan of the entire collection...
    master.WaitIdle();               # and wait until it completes.
    model.InvokePhiRegularizers();
    perplexity_score = master.GetScore(model, 'perplexity_score')
    print "Iter# = " + str(iter) + ", Perplexity = " + str(perplexity_score.value)

  top_tokens_score = master.GetScore(model, 'top_tokens_score')
  print top_tokens_score
