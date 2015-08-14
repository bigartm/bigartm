import uuid
import shutil
import tempfile
import itertools
import pytest

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants
import helpers

def test_func():
    # Set some constants
    num_tokens = 60
    num_items = 100
    pwt = 'pwt'
    nwt = 'nwt'
    
    num_topics = 10
    num_inner_iterations = 10
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

    batches_folder = tempfile.mkdtemp()
    try:
        # Generate small collection
        batch = messages.Batch()
        batch.id = str(uuid.uuid4())
        for token_id in xrange(num_tokens):
            batch.token.append('token_{0}'.format(token_id))

        for item_id in xrange(num_items):
            item = batch.item.add()
            item.id = item_id
            field = item.field.add()
            for token_id in xrange(num_tokens):
                field.token_id.append(token_id)
                background_count = ((item_id + token_id) % 5 + 1) if (token_id >= 40) else 0
                target_topics = num_topics if (token_id < 40) and ((token_id % 10) == (item_id % 10)) else 0
                field.token_count.append(background_count + target_topics)

        # Create the instance of low-level API and helper object
        lib = artm.wrapper.LibArtm()
        helper = helpers.TestHelper(lib)

        # Save batch on the disk
        lib.ArtmSaveBatch(batches_folder, batch)

        # Create master component and scores
        config = messages.TopTokensScoreConfig()
        config.num_tokens = num_top_tokens
        scores = [('PerplexityScore', messages.PerplexityScoreConfig()),
                  ('TopTokensScore', config)]
        master_id = helper.create_master_component(scores=scores)
        helper.master_id = master_id

        # Initialize model
        helper.initialize_model(pwt, num_topics, source_type='batches', disk_path=batches_folder)

        for iter in xrange(num_outer_iterations):
            # Invoke one scan of the collection and normalize Phi
            helper.process_batches(pwt, nwt, num_inner_iterations, batches_folder)
            helper.normalize_model(pwt, nwt)  

            # Retrieve and print perplexity score
            perplexity_score = helper.retrieve_score(pwt, 'PerplexityScore')
            assert abs(perplexity_score.value - expected_perplexity_value_on_iteration[iter]) < perplexity_tol
            print 'Iteration#{0} : Perplexity = {1:.3f}'.format(iter, perplexity_score.value)

        # Retrieve and print top tokens score
        top_tokens_score = helper.retrieve_score(pwt, 'TopTokensScore')

        print 'Top tokens per topic:'
        top_tokens_triplets = zip(top_tokens_score.topic_index, zip(top_tokens_score.token, top_tokens_score.weight))
        for topic_index, group in itertools.groupby(top_tokens_triplets, key=lambda (topic_index, _): topic_index):
            string = 'Topic#{0} : '.format(topic_index)
            for _, (token, weight) in group:
                assert abs(weight - expected_top_tokens_weight) < top_tokens_tol
                string += ' {0}({1:.3f})'.format(token, weight)
            print string
    finally:
        shutil.rmtree(batches_folder)
