# This example demonstrates the usage of coherence regularizer and coherenct score option in
# TopTokens and TopicKernel scores. Note, that you need kos collection dictionary with
# tokens co-occurences info to launch this script correctly (see examle02_parse_collection.py).

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

    # configure TopTokens score to count coherence
    top_tokens_score_config = artm.messages_pb2.TopTokensScoreConfig()
    top_tokens_score_config.cooccurrence_dictionary_name = dictionary.name()
    top_tokens_coherence_score = master.CreateTopTokensScore(config=top_tokens_score_config)

    # configure TopicKernel score to count coherence
    topic_kernel_score_config = artm.messages_pb2.TopicKernelScoreConfig()
    topic_kernel_score_config.probability_mass_threshold = 0.2
    topic_kernel_score_config.cooccurrence_dictionary_name = dictionary.name()
    topic_kernel_coherence_score = master.CreateTopicKernelScore(config=topic_kernel_score_config)

    # configure ImproveCoherence regularizer
    improve_coherence = master.CreateImproveCoherencePhiRegularizer(dictionary_name=dictionary.name())

    # Configure the model
    model = master.CreateModel(topics_count=15,
                               inner_iterations_count=30,
                               topic_names=["topic" + str(i) for i in range(0, 16)])
    model.EnableRegularizer(improve_coherence, 0.0001)

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
                print "Top tokens average coherence = %.3f" %\
                  top_tokens_coherence_score.GetValue(model).average_coherence
                print "Topic kernels average coherence = %.3f" %\
                  topic_kernel_coherence_score.GetValue(model).average_coherence
