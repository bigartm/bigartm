# This example demonstrates various ways of retrieving Phi matrix from BigARTM
# -*- coding: utf-8 -*-

import artm.messages_pb2
import artm.library

# Create master component and infer topic model
batches_disk_path = 'kos'
with artm.library.MasterComponent(disk_path=batches_disk_path) as master:
    topic_names = []
    for topic_index in range(0, 10):
        topic_names.append("Topic" + str(topic_index))
    model = master.CreateModel(topic_names=topic_names)

    for iteration in range(0, 2):
        master.InvokeIteration()
        master.WaitIdle()  # wait for all batches are processed
        model.Synchronize()  # synchronize model

    # The following code retrieves one topic at a time.
    # This avoids retrieving large topic models in a single protobuf message.
    print "Output p(w|t) values for the first few tokens (alphabetically) in each topic:"
    for topic_name in topic_names:
        topic_model = master.GetTopicModel(model=model, topic_names={topic_name})  # retrieve one column in Phi matrix
        print topic_model.topic_name[0],
        for i in range(0, 5):
            print topic_model.token[i], "%.5f" % topic_model.token_weights[i].value[0],
        print "..."
