# Copyright 2017, Additive Regularization of Topic Models.

from __future__ import print_function

import os
import tempfile
import shutil
import pytest

from six.moves import range

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
    rwt = 'rwt'
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'

    num_topics = 10
    num_document_passes = 10
    num_outer_iterations = 8

    smsp_phi_tau = -20.0
    smsp_theta_tau = -3.0

    perplexity_tol = 1.0
    expected_perp_col_value_on_iteration = {
        0: 6650.1,
        1: 2300.2,
        2: 1996.8,
        3: 1786.1,
        4: 1692.7,
        5: 1644.4,
        6: 1612.3,
        7: 1589.5
    }
    expected_perp_doc_value_on_iteration = {
        0: 6614.6,
        1: 2295.0,
        2: 1996.4,
        3: 1786.1,
        4: 1692.7,
        5: 1644.2,
        6: 1611.7,
        7: 1588.6
    }
    expected_perp_zero_words_on_iteration = {
        0: 494,
        1: 210,
        2: 24,
        3: 0,
        4: 2,
        5: 10,
        6: 28,
        7: 47
    }

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
        perplexity_config = messages.PerplexityScoreConfig()
        perplexity_config.model_type = constants.PerplexityScoreConfig_Type_UnigramCollectionModel
        perplexity_config.dictionary_name = dictionary_name
        
        scores = {'PerplexityDoc': messages.PerplexityScoreConfig(),
                  'PerplexityCol': perplexity_config}
        master = mc.MasterComponent(lib, scores=scores)

        # Create collection dictionary and import it
        master.gather_dictionary(dictionary_target_name=dictionary_name,
                                 data_path=batches_folder,
                                 vocab_file_path=os.path.join(data_path, vocab))

        # Configure basic regularizers
        master.create_regularizer(name='SmoothSparsePhi',
                                  config=messages.SmoothSparsePhiConfig(dictionary_name=dictionary_name),
                                  tau=0.0)
        master.create_regularizer(name='SmoothSparseTheta', config=messages.SmoothSparseThetaConfig(), tau=0.0)

        # Initialize model
        master.initialize_model(model_name=pwt,
                                topic_names=['topic_{}'.format(i) for i in range(num_topics)],
                                dictionary_name=dictionary_name)

        for iter in range(num_outer_iterations):
            # Invoke one scan of the collection, regularize and normalize Phi
            master.clear_score_cache()
            master.process_batches(pwt=pwt,
                                   nwt=nwt,
                                   num_document_passes=num_document_passes,
                                   batches_folder=batches_folder,
                                   regularizer_name=['SmoothSparseTheta'],
                                   regularizer_tau=[smsp_theta_tau])
            master.regularize_model(pwt, nwt, rwt, ['SmoothSparsePhi'], [smsp_phi_tau])
            master.normalize_model(pwt, nwt, rwt)  

            # Retrieve perplexity score
            perplexity_doc_score = master.get_score('PerplexityDoc')
            perplexity_col_score = master.get_score('PerplexityCol')

            # Assert and print scores
            string = 'Iter#{0}'.format(iter)
            string += ': Collection perp. = {0:.1f}'.format(perplexity_col_score.value)
            string += ', Document perp. = {0:.1f}'.format(perplexity_doc_score.value)
            string += ', Zero words = {0}'.format(perplexity_doc_score.zero_words)
            print(string)

            print(perplexity_col_score.value, expected_perp_col_value_on_iteration[iter])
            assert abs(perplexity_col_score.value - expected_perp_col_value_on_iteration[iter]) < perplexity_tol
            assert abs(perplexity_doc_score.value - expected_perp_doc_value_on_iteration[iter]) < perplexity_tol
            assert perplexity_doc_score.zero_words - expected_perp_zero_words_on_iteration[iter] == 0
    finally:
        shutil.rmtree(batches_folder)
