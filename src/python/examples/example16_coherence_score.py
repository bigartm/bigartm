# This example demonstrates the usage of coherence option in TopicKernel score. There're three steps:
# 1) learn topic model to find topic kernel tokens
# 2) re-parse the collection to find the cooc info about this tokens and form new dictionary
# 3) learn new topic model and count coherence

import artm.messages_pb2, artm.library, sys, glob, os
import shutil  # is need to remove folder with created dictionary and batches after experiment, not necessary

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
batches_disk_path = 'kos'
unique_tokens = artm.library.Library().LoadDictionary(os.path.join(batches_disk_path, 'dictionary'))

# Create master component and infer topic model
print 'Infer temporary model to find topic kernel tokens...'
with artm.library.MasterComponent() as master:
    master.config().processors_count = 2
    master.Reconfigure()
    dictionary = master.CreateDictionary(unique_tokens)

    # configure TopicKernel score to count coherence
    topic_kernel_score_config = artm.messages_pb2.TopicKernelScoreConfig()
    topic_kernel_score_config.probability_mass_threshold = 0.7
    topic_kernel_score_config.cooccurrence_dictionary_name = dictionary.name()
    topic_kernel_coherence_score = master.CreateTopicKernelScore(config=topic_kernel_score_config)

    # Configure the model
    model = master.CreateModel(topics_count=15,
                               inner_iterations_count=5,
                               topic_names=["topic" + str(i) for i in range(0, 16)])

    model.Initialize(dictionary)  # Setup initial approximation for Phi matrix.

    # Online algorithm with AddBatch()
    update_every = master.config().processors_count
    batches = glob.glob(batches_disk_path + "/*.batch")

    for iteration in range(0, 20):
        for batch_index, batch_filename in enumerate(batches):
            master.AddBatch(batch_filename=batch_filename)
            if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
                master.WaitIdle()  # wait for all batches are processed
                model.Synchronize(decay_weight=0.9, apply_weight=0.1)  # synchronize model

    all_kernel_tokens = topic_kernel_coherence_score.GetValue(model).kernel_tokens

# Parse collection
print 'OK. Create dictionary with cooc info...'
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos_cooc'
collection_name = 'kos'

collection_parser_config = artm.messages_pb2.CollectionParserConfig()
collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci

collection_parser_config.docword_file_path = data_folder + 'docword.' + collection_name + '.txt'
collection_parser_config.vocab_file_path = data_folder + 'vocab.' + collection_name + '.txt'
collection_parser_config.target_folder = target_folder
collection_parser_config.dictionary_file_name = 'dictionary_kernel_cooc'
    
tokens_set = set()
for tokens in all_kernel_tokens:
    for token in tokens.value:
        tokens_set.add(token);

print 'Overall number of kernel tokens: ' + str(len(tokens_set))
for token in tokens_set:
    collection_parser_config.cooccurrence_token.append(token)
collection_parser_config.gather_cooc = True
    
unique_tokens = artm.library.Library().ParseCollection(collection_parser_config)

print 'OK. Infer main model...'
# Create master component and infer topic model
with artm.library.MasterComponent() as master:
    master.config().processors_count = 2
    master.Reconfigure()
    print '...create a dictionary in master for it...'
    dictionary = master.CreateDictionary(unique_tokens)

    # configure TopicKernel score to count coherence
    topic_kernel_score_config = artm.messages_pb2.TopicKernelScoreConfig()
    topic_kernel_score_config.probability_mass_threshold = 0.7
    topic_kernel_score_config.cooccurrence_dictionary_name = dictionary.name()
    topic_kernel_coherence_score = master.CreateTopicKernelScore(config=topic_kernel_score_config)

    # Configure the model
    model = master.CreateModel(topics_count=15,
                               inner_iterations_count=5,
                               topic_names=["topic" + str(i) for i in range(0, 16)])

    model.Initialize(dictionary)  # Setup initial approximation for Phi matrix.

    # Online algorithm with AddBatch()
    print '... start processing...'
    update_every = master.config().processors_count
    batches = glob.glob(target_folder + "/*.batch")

    for iteration in range(0, 20):
        for batch_index, batch_filename in enumerate(batches):
            master.AddBatch(batch_filename=batch_filename)
            if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
                master.WaitIdle()  # wait for all batches are processed
                model.Synchronize(decay_weight=0.9, apply_weight=0.1)  # synchronize model
                print "Iter: " + str(iteration) + ", Topic kernels average coherence = %.3f" %\
                  topic_kernel_coherence_score.GetValue(model).average_coherence
    print 'Done.'
    shutil.rmtree(target_folder)
