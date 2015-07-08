# This example demonstrates options to initialize topic model.

import artm.messages_pb2
import artm.library

# Load one test batch
batches_disk_path = 'kos'

with artm.library.MasterComponent() as master:
    model = master.CreateModel(topics_count=10)
    model.config().use_new_tokens = False  # remember to disable auto-gathering of new tokens from batches
    model.Reconfigure()

    init_args = artm.messages_pb2.InitializeModelArgs()
    init_args.source_type = artm.library.InitializeModelArgs_SourceType_Batches
    init_args.disk_path = batches_disk_path
    init_filter = init_args.filter.add()
    init_filter.max_percentage = 0.2  # filter out frequent tokens present in 20% (or more) of items in the collection
    init_filter.min_items = 10  # filter out rare tokens that are present in up to 10 items

    # Use the following option to separately filter each modality.
    # By default filter is applied to all modalities.
    # init_filter.class_id = ... (for example "@default_class")

    # The following alternatives are also available, but they are not as useful as previous options.
    # init_filter.min_percentage = ...    # opposite to max_percentage
    # init_filter.max_items = ...         # opposite to min_items
    # init_filter.min_total_count = 50    # filter out tokens that have in total less than 50 occurrences in collection
    # init_filter.min_one_item_count = 5  # use tokens only if they are present 5 or more times in a single item

    model.Initialize(args=init_args)

    # At this point you may start running iterations
    # for iteration in range(0, 2):
    #     master.InvokeIteration(disk_path=batches_disk_path)
    #     master.WaitIdle()
    #     model.Synchronize()

    topic_model, numpy_matrix = master.GetTopicModel(model=model)
    print "Resulting topic model contains", len(topic_model.token), "tokens"
