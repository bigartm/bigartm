# This advanced example demonstrates how to upload existing Phi matrix back into BigARTM.
# There are several gotchas that you need to be aware about:
# 1. You should learn the difference between pwt-requests and nwt-requests.
#    This is explained in the comments further below.
# 2. ArtmOverwriteTopicModel only schedules the update of topic model.
#    To apply it you need to call WaitIdle() and Synchronize().
# 3. For pwt-requests you should use InvokeRegularizers=False in Synchronize() call at step#2.
#    For nwt-requests you should use InvokeRegularizers=True  in Synchronize() call at step#2.

import glob

import artm.messages_pb2
import artm.library

# Load one test batch
batches_disk_path = 'kos'
batches = glob.glob(batches_disk_path + "/*.batch")
test_batch = artm.library.Library().LoadBatch(batches[0])

# Pwt and Nwt requests represent two different ways of retrieving topic model.
# - Pwt represent a probability distribution p(w|t) over words for each topic.
#   This values are normalized so that sum_w p(w|t) = 1.
#   This data is sufficient to infer topic distribution p(t|d) for new documents,
#   however this data is not sufficient to continue tuning topic model
# - Nwt represent internal counters (n_wt) of topic model.
#   This data has the same layout as p(w|t) distribution,
#   e.g. there is one scalar n_wt value for each token and topic.
#   This values are not normalized, and does not include regularization coefficients.
#   However, by placing n_wt counters back into BigARTM you fully recover topic model state.
pwt_request = artm.library.GetTopicModelArgs_RequestType_Pwt
nwt_request = artm.library.GetTopicModelArgs_RequestType_Nwt

# Change request_type to nwt_request and re-run this example
# Then you will get the same p(t|d) distributions from 'master' and 'test_master'.
request_type = pwt_request

# Split 10 topics into two groups - even topics and odd topics.
topic_names = []
topic_names_odd = []
topic_names_even = []
for topic_index in range(0, 10):
    topic_name = "Topic" + str(topic_index)
    topic_names.append(topic_name)
    if topic_index % 2 == 0:
        topic_names_odd.append(topic_name)
    else:
        topic_names_even.append(topic_name)

# Create topic model in 'master', then upload it into 'test_master'.
with artm.library.MasterComponent() as master, artm.library.MasterComponent() as test_master:
    model = master.CreateModel(topic_names=topic_names)
    for iteration in range(0, 2):
        master.InvokeIteration(disk_path=batches_disk_path)
        master.WaitIdle()
        model.Synchronize()

    topic_model_odd = master.GetTopicModel(model=model, request_type=request_type, topic_names=topic_names_odd,
                                           use_matrix=False)
    topic_model_even = master.GetTopicModel(model=model, request_type=request_type, topic_names=topic_names_even,
                                            use_matrix=False)
    theta_matrix, numpy_matrix = master.GetThetaMatrix(model=model, batch=test_batch)

    print "Theta distribution for one test document: "
    print "For the original topic model:             ",
    for value in numpy_matrix[0, :]:
        print "%.5f" % value,

    test_model = test_master.CreateModel(topic_names=topic_names)
    test_model.Overwrite(topic_model_odd, commit=False)
    test_model.Overwrite(topic_model_even, commit=False)
    test_master.WaitIdle()
    invoke_regularizers = False if (request_type == pwt_request) else True
    test_model.Synchronize(decay_weight=0.0, apply_weight=1.0, invoke_regularizers=False)
    test_theta, numpy_matrix = test_master.GetThetaMatrix(model=test_model, batch=test_batch)
    print "\nFor topic model copied into test_master:  ",
    for value in numpy_matrix[0, :]:
        print "%.5f" % value,
    print '(the same result is expected)'

    # Continue topic model inference and compare new topic models
    master.InvokeIteration(disk_path=batches_disk_path)
    master.WaitIdle()
    model.Synchronize(decay_weight=0.5, apply_weight=1.0)
    theta_matrix, numpy_matrix = master.GetThetaMatrix(model=model, batch=test_batch)
    print "After updating original topic model:      ",
    for value in numpy_matrix[0, :]:
        print "%.5f" % value,

    test_master.InvokeIteration(disk_path=batches_disk_path)
    test_master.WaitIdle()
    test_model.Synchronize(decay_weight=0.5, apply_weight=1.0)
    test_theta, numpy_matrix = test_master.GetThetaMatrix(model=test_model, batch=test_batch)
    print "\nAfter updating topic model in test_master:",
    for value in numpy_matrix[0, :]:
        print "%.5f" % value,
    print '(depends on nwt vs pwt request)'
