# This example demonstrates two modes of perplexity calculation
# (UnigramDocumentModel and UnigramCollectionModel).
# Note that UnigamCollectionModel require a dictionary that stores frequency
# of every token in the collection. This dictionary must be created in master component
# with CreateDictionary() method. In this particular example
# the dictionary comes from the ParseCollectionOrLoadDictionary(),
# but you may use the dictionary from any other collection you have at your disposal.

import artm.messages_pb2, artm.library, sys

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'
unique_tokens = artm.library.Library().ParseCollectionOrLoadDictionary(
    data_folder + 'docword.' + collection_name + '.txt',
    data_folder + 'vocab.' + collection_name + '.txt',
    target_folder)

with artm.library.MasterComponent() as master:
    smsp_theta_reg = master.CreateSmoothSparseThetaRegularizer()
    smsp_phi_reg = master.CreateSmoothSparsePhiRegularizer()

    # Create dictionary with tokens frequencies
    dictionary = master.CreateDictionary(unique_tokens)

    # Default perplexity has type PerplexityScoreConfig_Type_UnigramDocumentModel
    perplexity_document_score = master.CreatePerplexityScore()

    # Create perplexity of type PerplexityScoreConfig_Type_UnigramCollectionModel
    perplexity_collection_config = artm.messages_pb2.PerplexityScoreConfig()
    perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramCollectionModel
    perplexity_collection_config.dictionary_name = unique_tokens.name
    perplexity_collection_score = master.CreatePerplexityScore(config=perplexity_collection_config)

    # Configure the model
    model = master.CreateModel(topics_count=10, inner_iterations_count=10)
    model.EnableScore(perplexity_document_score)
    model.EnableScore(perplexity_collection_score)
    model.EnableRegularizer(smsp_theta_reg, -1.0)
    model.EnableRegularizer(smsp_phi_reg, -1.0)
    model.Initialize(dictionary)       # Setup initial approximation for Phi matrix.

    for iteration in range(0, 8):
        master.InvokeIteration(disk_path=target_folder)  # Invoke one scan of the entire collection...
        master.WaitIdle()                                # and wait until it completes.
        model.Synchronize()                              # Synchronize topic model.
        perplexity_collection = perplexity_collection_score.GetValue(model)
        perplexity_document = perplexity_document_score.GetValue(model)
        print "Iter#" + str(iteration),
        print ": Collection perplexity = %.3f" % perplexity_collection.value,
        print ", Document perplexity = %.3f " % perplexity_document.value,
        print ", Zero words = %i " % perplexity_document.zero_words
