import random
import uuid
import os
import shutil
import tempfile
import itertools
import pytest

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants

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
        0: 55.620,
        1: 37.397,
        2: 28.466,
        3: 23.285,
        4: 20.741,
        5: 20.529,
        6: 20.472,
        7: 20.453,
        8: 20.454,
        9: 20.455
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

        # Create the instance of low-level API
        lib = artm.wrapper.LibArtm()

        # Save batch on the disk
        lib.ArtmSaveBatch(batches_folder, batch)

        # Create master component and add scores
        master_config = messages.MasterComponentConfig()

        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'PerplexityScore'
        ref_score_config.type = constants.ScoreConfig_Type_Perplexity
        ref_score_config.config = messages.PerplexityScoreConfig().SerializeToString()

        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'TopTokensScore'
        ref_score_config.type = constants.ScoreConfig_Type_TopTokens
        config = messages.TopTokensScoreConfig()
        config.num_tokens = num_top_tokens
        ref_score_config.config = config.SerializeToString()

        master_id = lib.ArtmCreateMasterComponent(master_config)

        # Initialize model
        init_args = messages.InitializeModelArgs()
        init_args.model_name = pwt
        init_args.disk_path = batches_folder
        init_args.source_type = constants.InitializeModelArgs_SourceType_Batches
        init_args.topics_count = num_topics
        lib.ArtmInitializeModel(master_id, init_args)

        # Create configuration for batch processing
        proc_args = messages.ProcessBatchesArgs()
        proc_args.pwt_source_name = pwt
        proc_args.nwt_target_name = nwt

        for name in os.listdir(batches_folder):
            proc_args.batch_filename.append(os.path.join(batches_folder, name))
        proc_args.inner_iterations_count = num_inner_iterations

        # Create configuration for Phi normalization
        norm_args = messages.NormalizeModelArgs()
        norm_args.pwt_target_name = pwt
        norm_args.nwt_source_name = nwt

        # Create config for scores retrieval
        perplexity_args = messages.GetScoreValueArgs()
        perplexity_args.model_name = pwt
        perplexity_args.score_name = 'PerplexityScore'

        top_tokens_args = messages.GetScoreValueArgs()
        top_tokens_args.model_name = pwt
        top_tokens_args.score_name = 'TopTokensScore'

        for iter in xrange(num_outer_iterations):
            # Invoke one scan of the collection and normalize Phi
            lib.ArtmRequestProcessBatches(master_id, proc_args)
            lib.ArtmNormalizeModel(master_id, norm_args)    

            # Retrieve and print perplexity score
            results = lib.ArtmRequestScore(master_id, perplexity_args)
            score_data = messages.ScoreData()
            score_data.ParseFromString(results)
            perplexity_score = messages.PerplexityScore()
            perplexity_score.ParseFromString(score_data.data)

            assert abs(perplexity_score.value - expected_perplexity_value_on_iteration[iter]) < perplexity_tol
            print 'Iteration#{0} : Perplexity = {1:.3f}'.format(iter, perplexity_score.value)

        # Retrieve and print top tokens score
        results = lib.ArtmRequestScore(master_id, top_tokens_args)
        score_data = messages.ScoreData()
        score_data.ParseFromString(results)
        top_tokens_score = messages.TopTokensScore()
        top_tokens_score.ParseFromString(score_data.data)

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
