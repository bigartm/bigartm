# -*- coding: utf-8 -*-
import os
import numpy
import tempfile
import shutil
import pytest

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants
import helpers

def test_func():
    # Set some constants
    dictionary_name = 'dictionary'
    pwt = 'pwt'
    nwt = 'nwt'
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'

    num_topics = 8
    num_outer_iterations = 2
    num_inner_iterations = 1
    
    theta_value = 0.1
    theta_tol = 0.1
    num_items = [1000, 430]
    pair_num_items = [1430, 2000]
    total_num_items = 3430

    batches_folder = tempfile.mkdtemp()
    try:
        # Create the instance of low-level API and helper object
        lib = artm.wrapper.LibArtm()
        helper = helpers.TestHelper(lib)

        # Parse collection from disk
        lib.ArtmParseCollection({'format': constants.CollectionParserConfig_Format_BagOfWordsUci,
                                 'docword_file_path': os.path.join(os.getcwd(), docword),
                                 'vocab_file_path': os.path.join(os.getcwd(), vocab),
                                 'target_folder': batches_folder,
                                 'dictionary_file_name': dictionary_name})

        # Create master component and scores
        scores = [('ThetaSnippetScore', messages.ThetaSnippetScoreConfig())]
        helper.master_id = helper.create_master_component(scores=scores, cache_theta=True)

        # Import the collection dictionary
        helper.import_dictionary(os.path.join(batches_folder, dictionary_name), dictionary_name)

        # Initialize model
        helper.initialize_model(pwt, num_topics, source_type='dictionary', dictionary_name=dictionary_name)

        for iter in xrange(num_outer_iterations):
            # Invoke one scan of the collection and normalize Phi
            helper.process_batches(pwt, nwt, num_inner_iterations, batches_folder)
            helper.normalize_model(pwt, nwt)

        # Option 1.
        # Getting a small snippet of ThetaMatrix for last processed documents (just to get an impression how it looks)
        # This may be useful if you are debugging some weird behavior, playing with regularizer weights, etc.
        # This does not require 'master.config().cache_theta = True'
        theta_snippet_score = helper.retrieve_score(pwt, 'ThetaSnippetScore')

        print 'Option 1. ThetaSnippetScore.'
        snippet_tuples = zip(theta_snippet_score.values, theta_snippet_score.item_id)
        print_string = ''
        for values, item_id in snippet_tuples:
            print_string += 'Item# {0}:\t'.format(item_id)
            for value in values.value:
                print_string += '{0:.3f}\t'.format(value)
                assert(abs(value - theta_value) < theta_tol)
            print print_string
            print_string = ''

        # Option 2.
        # Getting a full theta matrix cached during last iteration
        # This does requires "master_component.cache_theta = True" and stores the entire Theta matrix in memory.
        theta_matrix_info = helper.get_theta_info(model=pwt)
        theta_numpy_matrix = helper.get_theta_matrix(model=pwt, clean_cache=True)
        print_string = 'Option 2. Full ThetaMatrix cached during last iteration,'
        print_string += '#items = {0}'.format(len(theta_matrix_info.item_id))
        print print_string
        print theta_numpy_matrix
        assert numpy.count_nonzero(theta_numpy_matrix) == theta_numpy_matrix.size
        assert len(theta_matrix_info.item_id) == total_num_items

        # Option 3.
        # Getting theta matrix online during iteration.
        # This does requires "master_component.cache_theta = True", but never caches the entire Theta
        # because we clean it.
        # This is the best alternative to Option 2 if you can not afford caching entire ThetaMatrix in memory.
        batches = []
        for name in os.listdir(batches_folder):
            _, extension = os.path.splitext(name)
            if extension == '.batch':
                batches.append(os.path.join(batches_folder, name))
        for batch_index, batch_filename in enumerate(batches):
            helper.process_batches(pwt, nwt, num_inner_iterations, batches=[batch_filename])
            helper.normalize_model(pwt, nwt)

            # The following rule defines when to retrieve Theta matrix. You decide :)
            if ((batch_index + 1) % 2 == 0) or ((batch_index + 1) == len(batches)):
                theta_matrix_info = helper.get_theta_info(model=pwt)
                theta_numpy_matrix = helper.get_theta_matrix(model=pwt, clean_cache=True)
                print 'Option 3. ThetaMatrix from cache, online, #items = {0}'.format(len(theta_matrix_info.item_id))
                print theta_numpy_matrix
                assert numpy.count_nonzero(theta_numpy_matrix) == theta_numpy_matrix.size
                assert len(theta_matrix_info.item_id) in pair_num_items

        # Option 4.
        # Testing batches by explicitly loading them from disk. This is the right way of testing held-out batches.
        info, matrix = helper.process_batches(pwt=pwt,
                                              nwt=nwt,
                                              num_inner_iterations=1,
                                              batches=[batches[0]],
                                              find_theta=True)
        print 'Option 4. ThetaMatrix for test batch, #item {0}'.format(len(info.item_id))
        assert numpy.count_nonzero(matrix) == matrix.size
        assert len(info.item_id) in num_items
        print matrix
    finally:
        shutil.rmtree(batches_folder)
