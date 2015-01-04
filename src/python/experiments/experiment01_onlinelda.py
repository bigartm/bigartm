# BigARTM version: v0.5.5
#
# This example implements online algorithm where topic model is updated several times during one scan of the collection.
# The trick here is to frequently check the number of items processed so far, and synchronize topic model if the number
# of processed items exceeds certain threshold. The decay_weight and apply_weight in model.Synchronize() are set in
# the same way as in Online variational Bayes for LDA algorithm (Matthew D. Hoffman).
# (https://github.com/qpleple/online-lda-vb/blob/master/mdhoffma/onlinewikipedia.py)

# Quality of the model is controlled on held out test collection.
# To prepare train and test data for this example you should download the following archive:
# https://s3-eu-west-1.amazonaws.com/artm/enron_1k.7z
# Then unpack it into 'enron_train' folder, randomly select several .batch files and move them into
# 'D:\datasets\enron_test' folder.

import artm.messages_pb2, artm.library, sys, time, random, glob, math


def calc_phi_sparsity(topic_model):
    # topic_model should be an instance of messages_pb2.TopicModel class
    # (http://docs.bigartm.org/en/latest/ref/messages.html#messages_pb2.TopicModel)
    zeros = 0.0
    for token_index in range(0, len(topic_model.token_weights)):
        weights = topic_model.token_weights[token_index]
        for topic_index in range(0, len(weights.value)):
            if (weights.value[topic_index] < (0.001 / len(topic_model.token))):
                zeros += 1.0
    return zeros / (len(topic_model.token_weights) * topic_model.topics_count)


def calc_theta_sparsity(theta_matrix):
    # theta_matrix should be an instance of messages_pb2.ThetaMatrix class
    # (http://docs.bigartm.org/en/latest/ref/messages.html#messages_pb2.ThetaMatrix)
    zeros = 0.0
    for item_index in range(0, len(theta_matrix.item_weights)):
        weights = theta_matrix.item_weights[item_index]
        for topic_index in range(0, len(weights.value)):
            if (weights.value[topic_index] < (0.001 / numTopics)):
                zeros += 1.0
    return zeros / (len(theta_matrix.item_weights) * theta_matrix.topics_count)


def calc_perplexity(topic_model, theta_matrix, batch):
    item_map = {}
    token_map = {}
    perplexity = 0.0
    perplexity_norm = 0.0
    for item_index in range(0, len(theta_matrix.item_id)):
        item_map[theta_matrix.item_id[item_index]] = item_index
    for token_index in range(0, len(topic_model.token)):
        token_map[topic_model.token[token_index]] = token_index
    for item in batch.item:
        if not item.id in item_map:
            raise Exception('Unable to find item_id=' + str(item.id) + ' in the theta matrix')
        theta_item_index = item_map[item.id]
        item_weights = theta_matrix.item_weights[theta_item_index].value
        field = item.field[0]
        for field_token_index in range(0, len(field.token_id)):
            batch_token_index = field.token_id[field_token_index]
            token_count = field.token_count[field_token_index]
            token = batch.token[batch_token_index]
            if not token in token_map:
                raise Exception('Unable to find token=' + token + ' in the topic model')
            model_token_index = token_map[token]
            token_weights = topic_model.token_weights[model_token_index].value
            if len(token_weights) != len(item_weights):
                raise Exception('Inconsistent topics count between Phi and Theta matrices')
            pwd = 0.0
            for topic_index in range(0, len(token_weights)):
                pwd += token_weights[topic_index] * item_weights[topic_index]
            if pwd == 0:
                raise Exception('Implement DocumentUnigramModel or CollectionUnigramModel to resolve p(w|d)=0 cases')
            perplexity += token_count * math.log(pwd)
            perplexity_norm += token_count
    return perplexity, perplexity_norm

train_batches_folder = 'D:\\datasets\\enron_train\\'
test_batches_folder = 'D:\\datasets\\enron_test\\'

numTopics = 100

numProcessors = 1   # This value defines how many concurrent processors to use for calculation.

numInnerIters = 20  # Typical values of this parameter are between 10 and 50. The larger it is the better for
                    # convergence, but large values will increase runtime proportionally to this parameter.

batch_size = 1000   # Number of documents per batch.
                    # (!) IMPORTANT: Keep this value consistent with batches, stored in from train_batches_folder.
                    # Typical values:
                    # https://s3-eu-west-1.amazonaws.com/artm/enron_1k.7z            --- batch_size = 1000
                    # https://s3-eu-west-1.amazonaws.com/artm/nytimes_1k.7z          --- batch_size = 1000
                    # https://s3-eu-west-1.amazonaws.com/artm/pubmed_10k.7z          --- batch_size = 10000
                    # https://s3-eu-west-1.amazonaws.com/artm/enwiki-20141208_1k.7z  --- batch_size = 1000
                    # https://s3-eu-west-1.amazonaws.com/artm/enwiki-20141208_10k.7z --- batch_size = 10000

update_every = 1    # Synchronize model once per *update_every* batches

kappa = 0.5         # For kappa and tau0 refer to https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf
tau0 = 64           # (Online Learning for Latent Dirichlet Allocation, 'Matthew D. Hoffman')

alpha = 1.0 / numTopics
beta  = 1.0 / numTopics

unique_tokens = artm.library.Library().LoadDictionary(train_batches_folder + 'dictionary')

master_config = artm.messages_pb2.MasterComponentConfig()
master_config.processors_count = numProcessors
master_config.cache_theta = True
master_config.disk_path = train_batches_folder

perplexity_collection_config = artm.messages_pb2.PerplexityScoreConfig()
perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramCollectionModel
#perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramDocumentModel
perplexity_collection_config.dictionary_name = unique_tokens.name

# === TRAIN TOPIC MODEL ================================================================================================
with artm.library.MasterComponent(master_config) as master:
    dictionary = master.CreateDictionary(unique_tokens)
    perplexity_score = master.CreatePerplexityScore(config = perplexity_collection_config)
    smooth_sparse_phi = master.CreateSmoothSparsePhiRegularizer()
    smooth_sparse_theta = master.CreateSmoothSparseThetaRegularizer()

    items_processed_score = master.CreateItemsProcessedScore()

    # Configure the model
    model = master.CreateModel(config=artm.messages_pb2.ModelConfig(),
                               topics_count=numTopics, inner_iterations_count=numInnerIters)
    model.EnableScore(perplexity_score)
    model.EnableScore(items_processed_score)
    model.EnableRegularizer(smooth_sparse_phi, beta)
    model.EnableRegularizer(smooth_sparse_theta, alpha)

    model.Initialize(dictionary)    # Initialize random

    start_time = time.time()
    master.InvokeIteration(1)       # Invoke one scan of the entire collection

    done = False
    first_sync = True
    next_items_processed = (batch_size * update_every)
    while (not done):
        done = master.WaitIdle(10)    # Wait 10 ms and check if the number of processed items had changed
        current_items_processed = items_processed_score.GetValue(model).value
        if done or (current_items_processed >= next_items_processed):
            update_count = current_items_processed / (batch_size * update_every)
            next_items_processed = current_items_processed + (batch_size * update_every)      # set next model update
            rho = pow(tau0 + update_count, -kappa)                                            # calculate rho
            model.Synchronize(decay_weight=(0 if first_sync else (1-rho)), apply_weight=rho)  # synchronize model
            first_sync = False
            print "Items processed : %i " % current_items_processed,
            print "Accumulated perplexity : %.3f " % perplexity_score.GetValue(model).value,
            print "Elapsed time : %.3f " % (time.time() - start_time)

    print "Saving topic model... ",
    with open("Output.topic_model", "wb") as binary_file:
        binary_file.write(master.GetTopicModel(model).SerializeToString())
    print "Done. "

    # Perform one iteration to calculate Perplexity on the entire train dataset
    # master.InvokeIteration()
    # master.WaitIdle()
    # print "Train Perplexity calculated in BigARTM = %.3f" % perplexity_score.GetValue(model).value

# === TEST TOPIC MODEL =================================================================================================
test_master_config = artm.messages_pb2.MasterComponentConfig()
test_master_config.CopyFrom(master_config)
test_master_config.disk_path = test_batches_folder
with artm.library.MasterComponent(test_master_config) as test_master:
    print "Loading topic model... ",
    topic_model = artm.messages_pb2.TopicModel()
    with open("Output.topic_model", "rb") as binary_file:
        topic_model.ParseFromString(binary_file.read())
    print "Done. "

    test_dictionary = test_master.CreateDictionary(unique_tokens)
    test_perplexity_score = test_master.CreatePerplexityScore(config = perplexity_collection_config)
    smooth_sparse_phi = test_master.CreateSmoothSparsePhiRegularizer()
    smooth_sparse_theta = test_master.CreateSmoothSparseThetaRegularizer()

    test_model = test_master.CreateModel(topics_count = numTopics, inner_iterations_count = numInnerIters)
    test_model.EnableScore(test_perplexity_score)
    test_model.EnableRegularizer(smooth_sparse_phi, beta)
    test_model.EnableRegularizer(smooth_sparse_theta, alpha)
    test_model.Overwrite(topic_model)  # restore previously saved topic model into test_master

    print 'Estimate perplexity on held out batches... '
    perplexity = 0.0; perplexity_norm = 0.0
    for test_batch_filename in glob.glob(test_batches_folder + "*.batch"):
        test_batch = artm.library.Library().LoadBatch(test_batch_filename)
        test_batch_theta = test_master.GetThetaMatrix(model=test_model, batch=test_batch)
        theta_sparsity = calc_theta_sparsity(test_batch_theta)
        (batch_perplexity, batch_perplexity_norm) = calc_perplexity(topic_model, test_batch_theta, test_batch)
        print "Batch = " + test_batch_filename,
        print ", Theta sparsity = " + str(theta_sparsity),
        print ", Perplexity = " + str(math.exp(-batch_perplexity / batch_perplexity_norm))
        perplexity += batch_perplexity; perplexity_norm += batch_perplexity_norm
    print "Overall test perplexity = " + str(math.exp(-perplexity / perplexity_norm))

    test_master.InvokeIteration()
    test_master.WaitIdle()
    print "Test Perplexity calculated in BigARTM = %.3f" % test_perplexity_score.GetValue(test_model).value
