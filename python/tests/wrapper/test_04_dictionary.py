import os
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
    rwt = 'rwt'
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'

    num_topics = 10
    num_inner_iterations = 10
    num_outer_iterations = 8

    smsp_phi_tau = -20.0
    smsp_theta_tau = -3.0

    perplexity_tol = 0.1
    expected_perp_col_value_on_iteration = {
        0: 6716.3,
        1: 2310.1,
        2: 1997.8,
        3: 1786.1,
        4: 1692.8,
        5: 1644.7,
        6: 1613.2,
        7: 1591.0
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
        perplexity_config = messages.PerplexityScoreConfig()
        perplexity_config.model_type = constants.PerplexityScoreConfig_Type_UnigramCollectionModel
        perplexity_config.dictionary_name = dictionary_name
        
        scores = [('PerplexityDoc', messages.PerplexityScoreConfig()),
                  ('PerplexityCol', perplexity_config)]
        helper.master_id = helper.create_master_component(scores=scores)

        # Import the collection dictionary
        helper.import_dictionary(os.path.join(batches_folder, dictionary_name), dictionary_name)

        # Configure basic regularizers
        helper.create_smooth_sparse_phi_regularizer('SmoothSparsePhi', dictionary_name=dictionary_name)
        helper.create_smooth_sparse_theta_regularizer('SmoothSparseTheta')

        # Initialize model
        helper.initialize_model(pwt, num_topics, source_type='dictionary', dictionary_name=dictionary_name)

        for iter in xrange(num_outer_iterations):
            # Invoke one scan of the collection, regularize and normalize Phi
            helper.process_batches(pwt=pwt,
                                   nwt=nwt,
                                   num_inner_iterations=num_inner_iterations,
                                   batches_folder=batches_folder,
                                   regularizer_name=['SmoothSparseTheta'],
                                   regularizer_tau=[smsp_theta_tau])
            helper.regularize_model(pwt, nwt, rwt, ['SmoothSparsePhi'], [smsp_phi_tau])
            helper.normalize_model(pwt, nwt, rwt)  

            # Retrieve perplexity score
            perplexity_doc_score = helper.retrieve_score(pwt, 'PerplexityDoc')
            perplexity_col_score = helper.retrieve_score(pwt, 'PerplexityCol')

            # Assert and print scores
            string = 'Iter#{0}'.format(iter)
            string += ': Collection perp. = {0:.1f}'.format(perplexity_col_score.value)
            string += ', Document perp. = {0:.1f}'.format(perplexity_doc_score.value)
            string += ', Zero words = {0}'.format(perplexity_doc_score.zero_words)
            print string

            assert abs(perplexity_col_score.value - expected_perp_col_value_on_iteration[iter]) < perplexity_tol
            assert abs(perplexity_doc_score.value - expected_perp_doc_value_on_iteration[iter]) < perplexity_tol
            assert perplexity_doc_score.zero_words - expected_perp_zero_words_on_iteration[iter] == 0
    finally:
        shutil.rmtree(batches_folder)
