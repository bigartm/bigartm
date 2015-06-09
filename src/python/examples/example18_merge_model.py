# This example implements online algorithm with ProcessBatch(), MergeModel() and NormalizeModel().
# Topic model is updated several times during each scan of the collection.

import artm.messages_pb2, artm.library, sys, glob, os

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
    master.config().processors_count = 2
    master.Reconfigure()

    # Configure scores
    perplexity_score = master.CreatePerplexityScore()
    top_tokens_score = master.CreateTopTokensScore()

    # Initialize model
    pwt_model = "pwt"
    master.InitializeModel(model_name=pwt_model, batch_folder=target_folder, topics_count=10)

    # Perform iterations
    update_every = master.config().processors_count
    batches_to_process = []
    for iteration in range(0, 5):
        for batch_index, batch_filename in enumerate(batches):
            batches_to_process.append(batch_filename)
            if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
                master.ProcessBatches(pwt_model, batches_to_process, "nwt_hat")
                master.MergeModel({("nwt", 0.7), ("nwt_hat", 0.3)}, target_nwt="nwt")
                master.NormalizeModel("nwt", pwt_model)
                print "Iteration = %i," % iteration,
                print "Perplexity", batches_to_process, "= %.3f" % perplexity_score.GetValue(pwt_model).value
                batches_to_process = []

    # Visualize top token in each topic
    artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(pwt_model))