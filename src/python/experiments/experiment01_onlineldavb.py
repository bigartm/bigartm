# BigARTM version: v0.5.5
#
# This example implements online algorithm where topic model is updated several times during one scan of the collection.
# The trick here is to check every 10 ms the number of items processed so far, and synchronize topic model if the number
# of processed items exceeds certain threshold. The decay_weight and apply_weight in model.Synchronize() are set
# according to Online variational Bayes for LDA algorithm (Matthew D. Hoffman).

# Quality of the model is controlled on held out test collection.
# To prepare train and test data for this example you should download the following archive:
# https://s3-eu-west-1.amazonaws.com/artm/enron_1k.7z
# Then unpack it into 'enron_train' folder, randomly select several .batch files and move them into
# 'D:\datasets\enron_test' folder.

import artm.messages_pb2, artm.library, sys, time, random

train_batches_folder = 'D:\\datasets\\enron_train\\'
test_batches_folder = 'D:\\datasets\\enron_test\\'

numTopics = 32

numProcessors = 2   # This value defines how many concurrent processors to use for calculation.

numInnerIters = 20  # Typical values of this parameter are between 10 and 50. The larger it is the better for
                    # convergence, but large values will increase runtime proportionally to this parameter.

kappa = -0.5 # This parameter is introduced by Online variational Bayes for LDA algorithm. Recommended value is '-0.5'.

tau0 = 4     # This value defines controls how quickly to adjust 'rho' value in Online variational Bayes algorithm.
             # The 'rho' value will effectively change after (tau0 * S) documents. Please check if (tau0 * S) is of the
             # same order of magnitude as the overall size of your collection.

S = 1000     # This value defines after how many documents to update topic model in the online algorithm.
             # This value should not be smaller than numProcessors * numDocumentsPerBatch, where
             # numDocumentsPerBatch is equal to the number of items stored in each batch. For the 'enron' task,
             # downloaded from https://s3-eu-west-1.amazonaws.com/artm/enron_1k.7z, numDocumentsPerBatch = 500.

unique_tokens = artm.library.Library().LoadDictionary(train_batches_folder + 'dictionary')

master_config = artm.messages_pb2.MasterComponentConfig()
master_config.processors_count = numProcessors
master_config.cache_theta = True
master_config.disk_path = train_batches_folder

perplexity_collection_config = artm.messages_pb2.PerplexityScoreConfig()
perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramCollectionModel
perplexity_collection_config.dictionary_name = unique_tokens.name

# Create master component and infer topic model
with artm.library.MasterComponent(master_config) as master:
    dictionary = master.CreateDictionary(unique_tokens)
    perplexity_score = master.CreatePerplexityScore(config = perplexity_collection_config)

    items_processed_score = master.CreateItemsProcessedScore()

    # Configure the model
    model = master.CreateModel(config = artm.messages_pb2.ModelConfig(), topics_count = numTopics, inner_iterations_count = numInnerIters)
    model.EnableScore(perplexity_score)
    model.EnableScore(items_processed_score)
    model.Initialize(dictionary)    # Initialize random

    start_time = time.time()
    master.InvokeIteration(1)       # Invoke one scan of the entire collection

    done = False
    next_items_processed = S
    while (not done):
        done = master.WaitIdle(10)    # Wait 10 ms and check if the number of processed items had changed
        current_items_processed = items_processed_score.GetValue(model).value
        if done or (current_items_processed >= next_items_processed):
            next_items_processed = current_items_processed + S              # set next model update
            rho = pow(tau0 + current_items_processed / S, kappa)            # calculate decay_weight and apply_weight
            model.Synchronize(decay_weight=1-rho, apply_weight=rho)         # Synchronize topic model.
            print "Items processed : %i " % current_items_processed,
            print "Accumulated perplexity : %.3f " % perplexity_score.GetValue(model).value,
            print "Elapsed time : %.3f " % (time.time() - start_time)

    topic_model = master.GetTopicModel(model)   # retrieve Phi matrix
    theta_matrix = master.GetThetaMatrix(model) # retrieve Theta matrix

    # Calculate sparsity of the Phi matrix
    zeros = 0.0
    for token_index in range(0, len(topic_model.token_weights)):
        weights = topic_model.token_weights[token_index]
        for topic_index in range(0, len(weights.value)):
            if (weights.value[topic_index] < (0.001 / len(topic_model.token))):
                zeros += 1.0
    print "Sparsity of the Phi matrix : %.3f" % (zeros / (len(topic_model.token_weights) * numTopics))

    # Calculate sparsity of the Theta matrix on train data
    zeros = 0.0
    for item_index in range(0, len(theta_matrix.item_weights)):
        weights = theta_matrix.item_weights[item_index]
        for topic_index in range(0, len(weights.value)):
            if (weights.value[topic_index] < (0.001 / numTopics)):
                zeros += 1.0
    print "Sparsity of the Theta matrix on Train items: %.3f" % (zeros / (len(theta_matrix.item_weights) * numTopics))

    # Perform one more iteration to calculate Perplexity on the entire train dataset
    master.InvokeIteration()
    master.WaitIdle()
    print "Train Perplexity = %.3f" % perplexity_score.GetValue(model).value

# Now we would like to calculate perplexity of the model on held out datasets, stored in test_batches_folder.
# Ideally we would like to do so by passing test_batches_folder to master.InvokeIteration(), but this is not supported
# at the moment. An alternative is to create a new MasterComponent, and copy our topic model into it.
test_master_config = artm.messages_pb2.MasterComponentConfig()
test_master_config.CopyFrom(master_config)
test_master_config.disk_path = test_batches_folder
with artm.library.MasterComponent(test_master_config) as test_master:
    test_dictionary = test_master.CreateDictionary(unique_tokens)
    test_perplexity_score = test_master.CreatePerplexityScore(config = perplexity_collection_config)
    test_model = test_master.CreateModel(topics_count = numTopics, inner_iterations_count = numInnerIters)
    test_model.EnableScore(test_perplexity_score)
    test_model.Overwrite(topic_model)  # restore previously saved topic model into test_master
    test_master.InvokeIteration()
    test_master.WaitIdle()
    print "Test Perplexity = %.3f" % test_perplexity_score.GetValue(test_model).value

    # Retrieve and analyze test_theta_matrix (if you need this in your experiment)
    # test_theta_matrix = test_master.GetThetaMatrix(test_model)
