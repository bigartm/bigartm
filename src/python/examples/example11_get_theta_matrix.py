# This example demonstrates various ways of retrieving Theta matrix from BigARTM
# -*- coding: utf-8 -*-
import glob

import artm.messages_pb2
import artm.library

# Create master component and infer topic model
batches_disk_path = 'kos'
with artm.library.MasterComponent(disk_path=batches_disk_path) as master:
    master.config().cache_theta = True
    master.Reconfigure()

    model = master.CreateModel(topics_count=8)
    theta_snippet_score = master.CreateThetaSnippetScore()
    model.EnableScore(theta_snippet_score)

    for iteration in range(0, 2):
        master.InvokeIteration()
        master.WaitIdle()  # wait for all batches are processed
        model.Synchronize()  # synchronize model

    # Option 1.
    # Getting a small snippet of ThetaMatrix for last processed documents (just to get an impression how it looks)
    # This may be useful if you are debugging some weird behavior, playing with regularizer weights, etc.
    # This does not require "master.config().cache_theta = True"
    print "Option 1. ThetaSnippetScore."
    artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))

    # Option 2.
    # Getting a full theta matrix cached during last iteration
    # This does requires "master.config().cache_theta = True" and stores the entire Theta matrix in memory.
    theta_matrix = master.GetThetaMatrix(model, clean_cache=True)
    print "Option 2. Full ThetaMatrix cached during last iteration, #items = %i" % len(theta_matrix.item_id)

    # Option 3.
    # Getting theta matrix online during iteration.
    # This is the best alternative to Option 1 if you ca not afford caching entire ThetaMatrix in memory.
    batches = glob.glob(batches_disk_path + "/*.batch")
    for batch_index, batch_filename in enumerate(batches):
        master.AddBatch(batch_filename=batch_filename)

        # The following rule defines when to retrieve Theta matrix. You decide :)
        if ((batch_index + 1) % 2 == 0) or ((batch_index + 1) == len(batches)):
            master.WaitIdle()  # wait for all batches are processed
            # model.Synchronize(decay_weight=..., apply_weight=...)  # uncomment for online algorithm
            theta_matrix = master.GetThetaMatrix(model=model, clean_cache=True)
            print "Option 3. ThetaMatrix from cache, online, #items = %i" % len(theta_matrix.item_id)

    # Option 4.
    # Testing batches by explicitly loading them from disk. This is the right way of testing held-out batches.
    # This does not require "master.config().cache_theta = True"
    test_batch = artm.library.Library().LoadBatch(batches[0])  # select the first batch for demo purpose
    theta_matrix = master.GetThetaMatrix(model=model, batch=test_batch)
    print "Option 4. ThetaMatrix for test batch, #items = %i" % len(theta_matrix.item_id)
