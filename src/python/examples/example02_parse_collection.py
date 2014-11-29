# This example parses a small text collection from disk and process it with BigARTM.
# Topic model is configured with sparsity regularizers (for theta and phi matrices),
# and also with topic decorrelator. The weight of all regularizers was adjusted manually
# and then hardcoded in this script.
# Several scores are printed on every iteration (perplexity score, sparsity of theta and phi matrix).

import artm.messages_pb2, artm.library, sys, glob

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'

# The following code is the same as library.ParseCollectionOrLoadDictionary(),
# but it is important for you to understand what it does.
# Please learn ParseCollection() and LoadDictionary() methods.

batches_found = len(glob.glob(target_folder + "/*.batch"))
if batches_found == 0:
  print "No batches found, parsing them from textual collection...",
  collection_parser_config = artm.messages_pb2.CollectionParserConfig();
  collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci

  collection_parser_config.docword_file_path = data_folder + 'docword.'+ collection_name + '.txt'
  collection_parser_config.vocab_file_path = data_folder + 'vocab.'+ collection_name + '.txt'
  collection_parser_config.target_folder = target_folder
  collection_parser_config.dictionary_file_name = 'dictionary'
  unique_tokens = artm.library.Library().ParseCollection(collection_parser_config);
  print " OK."
else:
  print "Found " + str(batches_found) + " batches, using them."
  unique_tokens  = artm.library.Library().LoadDictionary(target_folder + '/dictionary');

# Create master component and infer topic model
with artm.library.MasterComponent(disk_path = target_folder) as master:
  # Create dictionary with tokens frequencies
  dictionary           = master.CreateDictionary(unique_tokens)

  # Configure basic scores
  perplexity_score     = master.CreatePerplexityScore()
  sparsity_theta_score = master.CreateSparsityThetaScore()
  sparsity_phi_score   = master.CreateSparsityPhiScore()
  top_tokens_score     = master.CreateTopTokensScore()
  theta_snippet_score  = master.CreateThetaSnippetScore()

  # Configure basic regularizers
  smsp_theta_reg   = master.CreateSmoothSparseThetaRegularizer()
  smsp_phi_reg     = master.CreateSmoothSparsePhiRegularizer()
  decorrelator_reg = master.CreateDecorrelatorPhiRegularizer()

  # Configure the model
  model = master.CreateModel(topics_count = 10, inner_iterations_count = 10)
  model.EnableScore(perplexity_score)
  model.EnableScore(sparsity_phi_score)
  model.EnableScore(sparsity_theta_score)
  model.EnableScore(top_tokens_score)
  model.EnableScore(theta_snippet_score)
  model.EnableRegularizer(smsp_theta_reg, -0.1)
  model.EnableRegularizer(smsp_phi_reg, -0.2)
  model.EnableRegularizer(decorrelator_reg, 1000000)
  model.Initialize(dictionary)       # Setup initial approximation for Phi matrix.

  for iter in range(0, 8):
    master.InvokeIteration(1)        # Invoke one scan of the entire collection...
    master.WaitIdle();               # and wait until it completes.
    model.Synchronize();             # Synchronize topic model.
    print "Iter#" + str(iter),
    print ": Perplexity = %.3f" % perplexity_score.GetValue(model).value,
    print ", Phi sparsity = %.3f" % sparsity_phi_score.GetValue(model).value,
    print ", Theta sparsity = %.3f" % sparsity_theta_score.GetValue(model).value

  artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(model))
  artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))
