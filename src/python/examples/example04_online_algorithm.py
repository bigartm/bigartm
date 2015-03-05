# This example implements online algorithm where topic model is updated several times
# during each scan of the collection.

import artm.messages_pb2, artm.library, sys, glob

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
batches_disk_path = 'kos'
unique_tokens = artm.library.Library().LoadDictionary(batches_disk_path + 'dictionary')

# Create master component and infer topic model
with artm.library.MasterComponent() as master:
    master.config().processors_count = 2
    master.Reconfigure()
    dictionary = master.CreateDictionary(unique_tokens)

    perplexity_score = master.CreatePerplexityScore()
    sparsity_theta_score = master.CreateSparsityThetaScore()
    sparsity_phi_score = master.CreateSparsityPhiScore()
    top_tokens_score = master.CreateTopTokensScore()
    theta_snippet_score = master.CreateThetaSnippetScore()

    # Configure basic regularizers
    theta_regularizer = master.CreateSmoothSparseThetaRegularizer()
    phi_regularizer = master.CreateSmoothSparsePhiRegularizer()
    decorrelator_regularizer = master.CreateDecorrelatorPhiRegularizer()

    # Configure the model
    model = master.CreateModel(topics_count=10, inner_iterations_count=10)
    model.EnableScore(perplexity_score)
    model.EnableScore(sparsity_phi_score)
    model.EnableScore(sparsity_theta_score)
    model.EnableScore(top_tokens_score)
    model.EnableScore(theta_snippet_score)
    model.EnableRegularizer(theta_regularizer, -0.1)
    model.EnableRegularizer(phi_regularizer, -0.2)
    model.EnableRegularizer(decorrelator_regularizer, 1000000)
    model.Initialize(dictionary)  # Setup initial approximation for Phi matrix.

    # Online algorithm with AddBatch()
    update_every = 4
    batches = glob.glob(batches_disk_path + "*.batch")
    for batch_index, batch_filename in enumerate(batches):
        master.AddBatch(batch_filename=batch_filename)
        if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
            master.WaitIdle()  # wait for all batches are processed
            model.Synchronize(decay_weight=0.9, apply_weight=0.1)  # synchronize model
            print "Perplexity = %.3f" % perplexity_score.GetValue(model).value,
            print ", Phi sparsity = %.3f " % sparsity_phi_score.GetValue(model).value,
            print ", Theta sparsity = %.3f" % sparsity_theta_score.GetValue(model).value

    artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(model))
    artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))
