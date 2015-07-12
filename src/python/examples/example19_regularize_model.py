# This example implements regularized offline algorithm with ProcessBatch(), RegularizeModel()and NormalizeModel().
# It demonstrates objective topics (with high sparsity) and background topics (without sparsity)

import artm.messages_pb2, artm.library, sys, glob

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'
if not glob.glob(target_folder + "/*.batch"):
    artm.library.Library().ParseCollection(
        docword_file_path=data_folder + 'docword.' + collection_name + '.txt',
        vocab_file_path=data_folder + 'vocab.' + collection_name + '.txt',
        target_folder=target_folder)

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

    theta_regularizers = {theta_objective.name() : -0.5, theta_background.name() : 0.5}
    phi_regularizers = {phi_objective.name() : -0.5, phi_background.name() : 0.5, decorrelator.name() : 100000}

    # Initialize model
    pwt_model = "pwt"
    master.InitializeModel(model_name=pwt_model, batch_folder=target_folder, topic_names=all_topics)

    # Perform iterations
    for iteration in range(0, 5):
        master.ProcessBatches(pwt_model, batches, "nwt", theta_regularizers, inner_iterations_count=30)
        master.RegularizeModel(pwt_model, "nwt", "rwt", phi_regularizers)
        master.NormalizeModel("nwt", pwt_model, "rwt")
        print "Perplexity = %.3f" % perplexity_score.GetValue(pwt_model).value,
        print ", Phi objective sparsity = %.3f" % sparsity_phi_objective.GetValue(pwt_model).value,
        print ", Theta objective sparsity = %.3f" % sparsity_theta_objective.GetValue(pwt_model).value

    # Visualize top token in each topic and a snippet of theta matrix
    artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(pwt_model))
    artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(pwt_model))
