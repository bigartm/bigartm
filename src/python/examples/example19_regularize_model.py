# This example implements regularized offline algorithm with ProcessBatch(), RegularizeModel()and NormalizeModel().
# It demonstrates objective topics (with high sparsity) and background topics (without sparsity)

import artm.messages_pb2, artm.library, sys, glob

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'
artm.library.Library().ParseCollectionOrLoadDictionary(
    data_folder + 'docword.' + collection_name + '.txt',
    data_folder + 'vocab.' + collection_name + '.txt',
    target_folder)

# Find file names of all batches in target folder
batches = glob.glob(target_folder + "/*.batch")

# Create master component
with artm.library.MasterComponent() as master:

    background_topics = []
    objective_topics = []
    all_topics = []

    for i in range(0, 16):
        topic_name = "topic" + str(i)
        all_topics.append(topic_name)
        if i < 14:
            objective_topics.append(topic_name)
        else:
            background_topics.append(topic_name)

    # Configure scores
    perplexity_score = master.CreatePerplexityScore()
    sparsity_theta_objective = master.CreateSparsityThetaScore(topic_names=objective_topics)
    sparsity_phi_objective = master.CreateSparsityPhiScore(topic_names=objective_topics)
    top_tokens_score = master.CreateTopTokensScore()
    theta_snippet_score = master.CreateThetaSnippetScore()

    # Configure regularizers
    theta_objective = master.CreateSmoothSparseThetaRegularizer(topic_names=objective_topics)
    theta_background = master.CreateSmoothSparseThetaRegularizer(topic_names=background_topics)
    phi_objective = master.CreateSmoothSparsePhiRegularizer(topic_names=objective_topics)
    phi_background = master.CreateSmoothSparsePhiRegularizer(topic_names=background_topics)
    decorrelator = master.CreateDecorrelatorPhiRegularizer(topic_names=objective_topics)

    theta_regularizers = {(theta_objective.name(), -0.5), (theta_background.name(), 0.5)}
    phi_regularizers = {(phi_objective.name(), -0.5), (phi_background.name(), 0.5), (decorrelator.name(), 100000)}

    # Initialize model
    master.InitializeModel(model_name="pwt", batch_folder=target_folder, topic_names=all_topics)

    # Perform iterations
    for iteration in range(0, 5):
        scores = master.ProcessBatches("pwt", batches, "nwt", theta_regularizers, inner_iterations_count=30)
        master.RegularizeModel("pwt", "nwt", "rwt", phi_regularizers)
        master.NormalizeModel("nwt", "pwt", "rwt")
        print "Perplexity = %.3f" % perplexity_score.GetValue(scores=scores).value,
        print ", Phi objective sparsity = %.3f" % sparsity_phi_objective.GetValue("pwt").value,
        print ", Theta objective sparsity = %.3f" % sparsity_theta_objective.GetValue(scores=scores).value

    # Visualize top token in each topic and a snippet of theta matrix
    artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue("pwt"))
    artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(scores=scores))
