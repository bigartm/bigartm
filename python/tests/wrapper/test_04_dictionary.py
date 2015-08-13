import os
import tempfile
import shutil
import pytest

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants

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
        0: 6694.7,
        1: 2317.9,
        2: 1985.1,
        3: 1772.9,
        4: 1662.5,
        5: 1608.5,
        6: 1580.9,
        7: 1563.1
    }
    expected_perp_doc_value_on_iteration = {
        0: 6602.3,
        1: 2312.9,
        2: 1985.0,
        3: 1772.6,
        4: 1662.1,
        5: 1607.5,
        6: 1578.8,
        7: 1560.5
    }
    expected_perp_zero_words_on_iteration = {
        0: 442,
        1: 70,
        2: 1,
        3: 4,
        4: 8,
        5: 20,
        6: 42,
        7: 53
    }

    batches_folder = tempfile.mkdtemp()
    try:
        # Create the instance of low-level API
        lib = artm.wrapper.LibArtm()
        
        # Parse collection from disk
        parser_config = messages.CollectionParserConfig()
        parser_config.format = constants.CollectionParserConfig_Format_BagOfWordsUci

        parser_config.docword_file_path = os.path.join(os.getcwd(), docword)
        parser_config.vocab_file_path = os.path.join(os.getcwd(), vocab)
        parser_config.target_folder = batches_folder
        parser_config.dictionary_file_name = dictionary_name

        lib.ArtmParseCollection(parser_config)

        # Create master component and add scores
        master_config = messages.MasterComponentConfig()

        # Add two instances of perplexity score
        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'PerplexityCollectionScore'
        ref_score_config.type = constants.ScoreConfig_Type_Perplexity
        
        perplexity_score = messages.PerplexityScoreConfig()
        perplexity_score.model_type = constants.PerplexityScoreConfig_Type_UnigramCollectionModel
        perplexity_score.dictionary_name = dictionary_name
        
        ref_score_config.config = perplexity_score.SerializeToString()

        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'PerplexityDocumentScore'
        ref_score_config.type = constants.ScoreConfig_Type_Perplexity
        ref_score_config.config = messages.PerplexityScoreConfig().SerializeToString()

        master_id = lib.ArtmCreateMasterComponent(master_config)

        # Configure sparse Phi regularizer
        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = 'SmoothSparsePhi'
        ref_reg_config.type = constants.RegularizerConfig_Type_SmoothSparsePhi

        sparse_phi_reg = messages.SmoothSparsePhiConfig()
        sparse_phi_reg.dictionary_name = dictionary_name

        ref_reg_config.config = sparse_phi_reg.SerializeToString()
        lib.ArtmCreateRegularizer(master_id, ref_reg_config)

        # Configure sparse Theta regularizer
        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = 'SmoothSparseTheta'
        ref_reg_config.type = constants.RegularizerConfig_Type_SmoothSparseTheta
        ref_reg_config.config = messages.SmoothSparseThetaConfig().SerializeToString()
        lib.ArtmCreateRegularizer(master_id, ref_reg_config)

        # Import the collection dictionary
        dict_args = messages.ImportDictionaryArgs()
        dict_args.dictionary_name = 'dictionary'
        dict_args.file_name = os.path.join(batches_folder, dictionary_name)
        lib.ArtmImportDictionary(master_id, dict_args)

        # Initialize model
        init_args = messages.InitializeModelArgs()
        init_args.model_name = pwt
        init_args.dictionary_name = dictionary_name
        init_args.source_type = constants.InitializeModelArgs_SourceType_Dictionary
        init_args.topics_count = num_topics
        lib.ArtmInitializeModel(master_id, init_args)

        # Create configuration for batch processing
        proc_args = messages.ProcessBatchesArgs()
        proc_args.regularizer_name.append('SmoothSparseTheta')
        proc_args.regularizer_tau.append(smsp_theta_tau)
        proc_args.pwt_source_name = pwt
        proc_args.nwt_target_name = nwt
        for name in os.listdir(batches_folder):
            if name != dictionary_name:
                proc_args.batch_filename.append(os.path.join(batches_folder, name))
        proc_args.inner_iterations_count = num_inner_iterations

        # Create configuration for Phi normalization
        norm_args = messages.NormalizeModelArgs()
        norm_args.pwt_target_name = pwt
        norm_args.nwt_source_name = nwt
        norm_args.rwt_source_name = rwt

        # Create configuration for Phi regularization
        reg_args = messages.RegularizeModelArgs()
        reg_args.pwt_source_name = pwt
        reg_args.nwt_source_name = nwt
        reg_args.rwt_target_name = rwt
        
        reg_set = reg_args.regularizer_settings.add()
        reg_set.name = 'SmoothSparsePhi'
        reg_set.tau = smsp_phi_tau
        reg_set.use_relative_regularization = False

        # Create config for scores retrieval
        perplexity_col_args = messages.GetScoreValueArgs()
        perplexity_col_args.model_name = pwt
        perplexity_col_args.score_name = 'PerplexityCollectionScore'

        perplexity_doc_args = messages.GetScoreValueArgs()
        perplexity_doc_args.model_name = pwt
        perplexity_doc_args.score_name = 'PerplexityDocumentScore'

        for iter in xrange(num_outer_iterations):
            # Invoke one scan of the collection, regularize and normalize Phi
            lib.ArtmRequestProcessBatches(master_id, proc_args)
            lib.ArtmRegularizeModel(master_id, reg_args)
            lib.ArtmNormalizeModel(master_id, norm_args)    

            # Retrieve perplexity score
            results = lib.ArtmRequestScore(master_id, perplexity_doc_args)
            score_data = messages.ScoreData()
            score_data.ParseFromString(results)
            perplexity_doc_score = messages.PerplexityScore()
            perplexity_doc_score.ParseFromString(score_data.data)

            results = lib.ArtmRequestScore(master_id, perplexity_col_args)
            score_data = messages.ScoreData()
            score_data.ParseFromString(results)
            perplexity_col_score = messages.PerplexityScore()
            perplexity_col_score.ParseFromString(score_data.data)

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
