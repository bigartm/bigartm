# This example implements online algorithm where topic model is updated several times
# during each scan of the collection.
# The trick here is to check number of processed items every 10ms,
# if synchronize topic model if the number had changed.
# The decay_weight is set to 0.75 because we expect 4 batches on 'kos' dataset.

import artm.messages_pb2, artm.library, sys

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'
unique_tokens = artm.library.Library().ParseCollectionOrLoadDictionary(
  data_folder + 'docword.'+ collection_name + '.txt',
  data_folder + 'vocab.' + collection_name + '.txt',
  target_folder)

# Create master component and infer topic model
with artm.library.MasterComponent(disk_path = target_folder) as master:
  dictionary           = master.CreateDictionary(unique_tokens)

  perplexity_score     = master.CreatePerplexityScore()
  sparsity_theta_score = master.CreateSparsityThetaScore()
  sparsity_phi_score   = master.CreateSparsityPhiScore()
  top_tokens_score     = master.CreateTopTokensScore()
  theta_snippet_score  = master.CreateThetaSnippetScore()
  items_processed_score = master.CreateItemsProcessedScore()

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
  model.EnableScore(items_processed_score)
  model.EnableRegularizer(smsp_theta_reg, -0.1)
  model.EnableRegularizer(smsp_phi_reg, -0.2)
  model.EnableRegularizer(decorrelator_reg, 1000000)
  model.Initialize(dictionary)    # Setup initial approximation for Phi matrix.

  for iter in range(0, 8):
    print "Iter#" + str(iter) + ":\n",
    master.InvokeIteration(1)       # Invoke one scan of the entire collection

    done = False
    items_processed = 0
    while (not done):
      done = master.WaitIdle(10);   # Wait 10 ms and check if the number of processed items had changed
      current_items_processed = items_processed_score.GetValue(model).value
      if (items_processed != current_items_processed):
        items_processed = current_items_processed
        model.Synchronize(0.75);         # Synchronize topic model.
        print "Perplexity = %.3f" % perplexity_score.GetValue(model).value,
        print ", Phi sparsity = %.3f " % sparsity_phi_score.GetValue(model).value,
        print ", Theta sparsity = %.3f" % sparsity_theta_score.GetValue(model).value

  artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(model))
  artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))
