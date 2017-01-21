# Copyright 2017, Additive Regularization of Topic Models.

from __future__ import print_function

import os
import itertools
import tempfile
import shutil
import pytest

from six.moves import range, zip

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants
import artm.master_component as mc

def test_func():
    # Set some constants
    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    dictionary_name = 'dictionary'
    pwt = 'pwt'
    nwt = 'nwt'
    nwt_hat = 'nwt_hat'
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'

    num_topics = 10
    num_document_passes = 10
    num_outer_iterations = 8
    num_processors = 2
    
    decay_weight = 0.7
    apply_weight = 0.3
    
    num_batches = 2
    top_tokens_value = 0.5
    top_tokens_tol = 0.5
    perplexity_first_value = set([6714.673, 6710.324, 6706.906, 6710.120, 6710.327, 6717.755,
                                  6717.757, 6698.847, 6710.120, 6714.667, 6698.852, 6706.903])

    batches_folder = tempfile.mkdtemp()
    try:
        # Create the instance of low-level API and master object
        lib = artm.wrapper.LibArtm()
        
        # Parse collection from disk
        lib.ArtmParseCollection({'format': constants.CollectionParserConfig_CollectionFormat_BagOfWordsUci,
                                 'docword_file_path': os.path.join(data_path, docword),
                                 'vocab_file_path': os.path.join(data_path, vocab),
                                 'target_folder': batches_folder})

        # Create master component and scores
        scores = {'Perplexity': messages.PerplexityScoreConfig(),
                  'TopTokens': messages.TopTokensScoreConfig()}
        master = mc.MasterComponent(lib, num_processors=num_processors, scores=scores)

        # Create collection dictionary and import it
        master.gather_dictionary(dictionary_target_name=dictionary_name,
                                 data_path=batches_folder,
                                 vocab_file_path=os.path.join(data_path, vocab))

        # Initialize model
        master.initialize_model(model_name=pwt,
                                topic_names=['topic_{}'.format(i) for i in range(num_topics)],
                                dictionary_name=dictionary_name)

        # Get file names of batches to process
        batches = []
        for name in os.listdir(batches_folder):
            _, extension = os.path.splitext(name)
            if extension == '.batch':
                batches.append(os.path.join(batches_folder, name))

        # Perform iterations
        update_every = num_processors
        batches_to_process = []
        for iter in range(num_outer_iterations):
            for batch_index, batch_filename in enumerate(batches):
                batches_to_process.append(batch_filename)
                if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
                    master.clear_score_cache()
                    master.process_batches(pwt, nwt_hat, num_document_passes, batches=batches_to_process)
                    master.merge_model({nwt: decay_weight, nwt_hat: apply_weight}, nwt=nwt)
                    master.normalize_model(pwt, nwt)

                    # Retrieve and print perplexity score
                    perplexity_score = master.get_score('Perplexity')
                    if iter == 0 and batch_index == 0:
                        assert(perplexity_score.value in perplexity_first_value)
                    assert len(batches_to_process) == num_batches
                    print_string = 'Iteration = {0},'.format(iter)
                    print_string += 'Perplexity = {0:.3f}'.format(perplexity_score.value)
                    print_string += ', num batches = {0}'.format(len(batches_to_process))
                    print(print_string)
                    batches_to_process = []

        # Retrieve and print top tokens score
        top_tokens_score = master.get_score('TopTokens')

        print('Top tokens per topic:')
        top_tokens_triplets = zip(top_tokens_score.topic_index, zip(top_tokens_score.token, top_tokens_score.weight))
        for topic_index, group in itertools.groupby(top_tokens_triplets, key=lambda triplet: triplet[0]):
            print_string = 'Topic#{0} : '.format(topic_index)
            for _, (token, weight) in group:
                print_string += ' {0}({1:.3f})'.format(token, weight)
                assert abs(weight - top_tokens_value) < top_tokens_tol
            print(print_string)
    finally:
        shutil.rmtree(batches_folder)
