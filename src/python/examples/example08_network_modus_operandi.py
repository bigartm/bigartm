# This example demonstrates network modus operandi in master component.

import artm.messages_pb2
import artm.library

# Set the following value to false if you are already running remote node_controller process
create_local_node_controller = True
if create_local_node_controller:
    node_controller1 = artm.library.NodeController('tcp://*:5555')
    node_controller2 = artm.library.NodeController('tcp://*:5556')

target_folder = 'kos'  # network path of a shared folder with batches to process.
# The folder must be reachable from all remote node controllers.

# load tokens from from local dictionary file
unique_tokens = artm.library.Library().LoadDictionary('kos/dictionary')

master_config = artm.messages_pb2.MasterComponentConfig()
master_config.modus_operandi = artm.library.MasterComponentConfig_ModusOperandi_Network
master_config.disk_path = target_folder
master_config.create_endpoint = 'tcp://*:5550'
master_config.connect_endpoint = 'tcp://localhost:5550'
master_config.node_connect_endpoint.append('tcp://localhost:5555')
master_config.node_connect_endpoint.append('tcp://localhost:5556')

with artm.library.MasterComponent(config=master_config) as master:
    dictionary = master.CreateDictionary(unique_tokens)
    perplexity_score = master.CreatePerplexityScore()
    model = master.CreateModel(topics_count=10, inner_iterations_count=10)
    model.EnableScore(perplexity_score)
    model.Initialize(dictionary)

    for iteration in range(0, 8):
        master.InvokeIteration(1)        # Invoke one scan of the entire collection...
        master.WaitIdle()                # and wait until it completes.
        model.Synchronize()              # Synchronize topic model.
        print "Iter#" + str(iteration),
        print ": Perplexity = %.3f" % perplexity_score.GetValue(model).value

if create_local_node_controller:
    node_controller1.Dispose()
    node_controller2.Dispose()
