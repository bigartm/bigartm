import sys, glob, random
import artm.messages_pb2, artm.library

# Parse collection
batches_found = len(glob.glob("kos/*.batch"))
artm_library = artm.library.Library()
if batches_found == 0:
  print "No batches found, parsing them from textual collection...",
  collection_parser_config = artm.messages_pb2.CollectionParserConfig();
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
with artm.library.MasterComponent(disk_path = 'kos') as master:
  perplexity_score     = master.CreatePerplexityScore()
  sparsity_theta_score = master.CreateSparsityThetaScore()
  sparsity_phi_score   = master.CreateSparsityPhiScore()
  top_tokens_score     = master.CreateTopTokensScore()
  theta_snippet_score  = master.CreateThetaSnippetScore()

  # Configure basic regularizers
  dirichlet_theta_reg  = master.CreateDirichletThetaRegularizer()
  dirichlet_phi_reg    = master.CreateDirichletPhiRegularizer()
  decorrelator_reg     = master.CreateDecorrelatorPhiRegularizer()

  # Configure the model
  model = master.CreateModel(topics_count = 10, inner_iterations_count = 10)
  model.EnableScore(perplexity_score)
  model.EnableScore(sparsity_phi_score)
  model.EnableScore(sparsity_theta_score)
  model.EnableScore(top_tokens_score)
  model.EnableScore(theta_snippet_score)
  model.EnableRegularizer(dirichlet_theta_reg, -0.1)
  model.EnableRegularizer(dirichlet_phi_reg, -0.2)
  model.EnableRegularizer(decorrelator_reg, 1000000)
  model.Initialize(unique_tokens)

  for iter in range(0, 8):
    master.InvokeIteration(1)        # Invoke one scan of the entire collection...
    master.WaitIdle();               # and wait until it completes.
    model.Synchronize();             # Synchronize topic model.
    print "Iter#" + str(iter),
    print ": Perplexity = %.3f" % perplexity_score.GetValue(model).value,
    print ", Phi sparsity = %.3f " % sparsity_phi_score.GetValue(model).value,
    print ", Theta sparsity = %.3f" % sparsity_theta_score.GetValue(model).value

  print '\nTop tokens per topic:'
  top_tokens_score = top_tokens_score.GetValue(model)
  for i in range(0, len(top_tokens_score.values)):
    print "Topic#" + str(i+1) + ": ",
    for value in top_tokens_score.values[i].value:
      print value + " ",
    print "\n",

  print '\nSnippet of theta matrix:'
  theta_snippet_score = theta_snippet_score.GetValue(model)
  for i in range(0, len(theta_snippet_score.values)):
    print "Item#" + str(theta_snippet_score.item_id[i]) + ": ",
    for value in theta_snippet_score.values[i].value:
      print "%.3f\t" % value,
    print "\n",
