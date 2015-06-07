# This example implements simple offline algorithm with ProcessBatch() and NormalizeModel().


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
    master.config().processors_count = 2
    master.Reconfigure()

    # Configure scores
    perplexity_score = master.CreatePerplexityScore()
    top_tokens_score = master.CreateTopTokensScore()

    # Initialize model
    master.InitializeModel(model_name="pwt", batch_folder=target_folder, topics_count=10)

    # Perform iterations
    for iteration in range(0, 5):
        scores = master.ProcessBatches("pwt", batches, "nwt")
        master.NormalizeModel("nwt", "pwt")
        print "Iteration = %i," % iteration, "Perplexity = %.3f" % perplexity_score.GetValue(scores=scores).value

    # Visualize top token in each topic
    artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue("pwt"))
