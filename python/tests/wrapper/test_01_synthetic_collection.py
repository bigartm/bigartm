# Copyright 2017, Additive Regularization of Topic Models.

from __future__ import print_function

import uuid
import shutil
import tempfile
import itertools
import pytest

from six.moves import range, zip

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.master_component as mc

def test_func():
    # Set some constants
    num_tokens = 60
    num_items = 100
    pwt = 'pwt'
    nwt = 'nwt'
    
    num_topics = 10
    num_document_passes = 10
    num_outer_iterations = 10
    num_top_tokens = 4

    perplexity_tol = 0.001
    expected_perplexity_value_on_iteration = {
        0: 54.616,
        1: 38.472,
        2: 28.655,
        3: 24.362,
        4: 22.355,
        5: 21.137,
        6: 20.808,
        7: 20.791,
        8: 20.746,
        9: 20.581
    }

    top_tokens_tol = 0.05
    expected_top_tokens_weight = 0.1

    dictionary_name = 'dictionary'
    batches_folder = tempfile.mkdtemp()
    try:
        # Generate small collection
        batch = messages.Batch()
        batch.id = str(uuid.uuid4())
        for token_id in range(num_tokens):
            batch.token.append('token_{0}'.format(token_id))

        for item_id in range(num_items):
            item = batch.item.add()
            item.id = item_id
            for token_id in range(num_tokens):
                item.token_id.append(token_id)
                background_count = ((item_id + token_id) % 5 + 1) if (token_id >= 40) else 0
                target_topics = num_topics if (token_id < 40) and ((token_id % 10) == (item_id % 10)) else 0
                item.token_weight.append(background_count + target_topics)

        # Create the instance of low-level API
        lib = artm.wrapper.LibArtm()

        # Save batch on the disk
        lib.ArtmSaveBatch(batches_folder, batch)

        # Create master component and scores
        scores = {'PerplexityScore': messages.PerplexityScoreConfig(),
                  'TopTokensScore': messages.TopTokensScoreConfig(num_tokens = num_top_tokens)}
        master = mc.MasterComponent(lib, scores=scores)

        # Create collection dictionary and import it
        master.gather_dictionary(dictionary_target_name=dictionary_name, data_path=batches_folder)

        # Initialize model
        master.initialize_model(model_name=pwt,
                                topic_names=['topic_{}'.format(i) for i in range(num_topics)],
                                dictionary_name=dictionary_name)

        for iter in range(num_outer_iterations):
            # Invoke one scan of the collection and normalize Phi
            master.clear_score_cache()
            master.process_batches(pwt, nwt, num_document_passes, batches_folder)
            master.normalize_model(pwt, nwt)

            # Retrieve and print perplexity score
            perplexity_score = master.get_score('PerplexityScore')
            assert abs(perplexity_score.value - expected_perplexity_value_on_iteration[iter]) < perplexity_tol
            print('Iteration#{0} : Perplexity = {1:.3f}'.format(iter, perplexity_score.value))

        # Retrieve and print top tokens score
        top_tokens_score = master.get_score('TopTokensScore')

        print('Top tokens per topic:')
        top_tokens_triplets = zip(top_tokens_score.topic_index, zip(top_tokens_score.token, top_tokens_score.weight))
        for topic_index, group in itertools.groupby(top_tokens_triplets, key=lambda triplet: triplet[0]):
            string = 'Topic#{0} : '.format(topic_index)
            for _, (token, weight) in group:
                assert abs(weight - expected_top_tokens_weight) < top_tokens_tol
                string += ' {0}({1:.3f})'.format(token, weight)
            print(string)
    finally:
        shutil.rmtree(batches_folder)
