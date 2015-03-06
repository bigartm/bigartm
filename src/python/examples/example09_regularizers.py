# This example demonstrates objective topics (with high sparsity) and background topics (without sparsity)

import artm.messages_pb2, artm.library, sys, glob, os

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
batches_disk_path = 'kos'
unique_tokens = artm.library.Library().LoadDictionary(os.path.join(batches_disk_path, 'dictionary'))

# Create master component and infer topic model
with artm.library.MasterComponent() as master:
    master.config().processors_count = 2
    master.Reconfigure()
    dictionary = master.CreateDictionary(unique_tokens)

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

    perplexity_score = master.CreatePerplexityScore()
    sparsity_theta_objective = master.CreateSparsityThetaScore(topic_names=objective_topics)
    sparsity_phi_objective = master.CreateSparsityPhiScore(topic_names=objective_topics)
    top_tokens_score = master.CreateTopTokensScore()
    theta_snippet_score = master.CreateThetaSnippetScore()

    # Configure basic regularizers
    theta_objective = master.CreateSmoothSparseThetaRegularizer(topic_names=objective_topics)
    theta_background = master.CreateSmoothSparseThetaRegularizer(topic_names=background_topics)
    phi_objective = master.CreateSmoothSparsePhiRegularizer(topic_names=objective_topics)
    phi_background = master.CreateSmoothSparsePhiRegularizer(topic_names=background_topics)
    decorrelator_regularizer = master.CreateDecorrelatorPhiRegularizer(topic_names=objective_topics)

    # Configure the model
    model = master.CreateModel(topics_count=10, inner_iterations_count=30, topic_names=all_topics)
    model.EnableScore(perplexity_score)
    model.EnableScore(sparsity_theta_objective)
    model.EnableScore(sparsity_phi_objective)
    model.EnableScore(top_tokens_score)
    model.EnableScore(theta_snippet_score)
    model.EnableRegularizer(theta_objective, -1.0)
    model.EnableRegularizer(theta_background, 0.5)
    model.EnableRegularizer(phi_objective, -1.0)
    model.EnableRegularizer(phi_background, 1.0)
    model.EnableRegularizer(decorrelator_regularizer, 1000000)
    model.Initialize(dictionary)  # Setup initial approximation for Phi matrix.

    # Online algorithm with AddBatch()
    update_every = master.config().processors_count
    batches = glob.glob(batches_disk_path + "/*.batch")

    for iteration in range(0, 5):
        for batch_index, batch_filename in enumerate(batches):
            master.AddBatch(batch_filename=batch_filename)
            if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
                master.WaitIdle()  # wait for all batches are processed
                model.Synchronize(decay_weight=0.9, apply_weight=0.1)  # synchronize model
                print "Perplexity = %.3f" % perplexity_score.GetValue(model).value,
                print ", Phi objective sparsity = %.3f" % sparsity_phi_objective.GetValue(model).value,
                print ", Theta objective sparsity = %.3f" % sparsity_theta_objective.GetValue(model).value

    artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(model))
    artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))
