# This example prints out the average execution time per iteration
# depending on the number of concurrent processors.

import artm.messages_pb2
import artm.library
import sys
import time

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'
unique_tokens = artm.library.Library().ParseCollectionOrLoadDictionary(
    data_folder + 'docword.' + collection_name + '.txt',
    data_folder + 'vocab.' + collection_name + '.txt',
    target_folder)

# Create master component and infer topic model
for processors_count in [4, 2, 1]:
    with artm.library.MasterComponent() as master:
        dictionary = master.CreateDictionary(unique_tokens)

        perplexity_score = master.CreatePerplexityScore()
        model = master.CreateModel(topics_count=10, inner_iterations_count=10)
        model.Initialize(dictionary)       # Setup initial approximation for Phi matrix.

        print "Setting processors_count to " + str(processors_count)
        master.config().processors_count = processors_count
        master.Reconfigure()

        times = []
        num_iters = 5
        for iteration in range(0, num_iters):
            start = time.time()
            master.InvokeIteration(disk_path=target_folder)  # Invoke one scan over all batches,
            master.WaitIdle()                                # and wait until it completes.
            model.Synchronize()                              # Synchronize topic model.
            end = time.time()
            times.append(end - start)
            print "Iter#" + str(iteration),
            print ": Perplexity = %.3f" % perplexity_score.GetValue(model).value + ", Time = %.3f" % (end - start)

        print "Averate time per iteration = %.3f " % (float(sum(times)) / len(times)) + "\n"
