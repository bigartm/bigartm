import os
import glob
import itertools
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

    smsp_phi_tau = -0.2
    smsp_theta_tau = -0.1
    decor_phi_tau = 1000000

    num_topics = 10
    inner_iterations_count = 10
    outer_iterations_count = 8

    tolerance = 0.001
    expected_perplexity_value_on_iteration = {
        0: 6722.401,
        1: 2452.129,
        2: 2375.061,
        3: 1830.576,
        4: 1760.534,
        5: 1659.460,
        6: 1638.274,
        7: 1604.741,
    }
    expected_phi_sparsity_value_on_iteration = {
        0: 0.059,
        1: 0.119,
        2: 0.211,
        3: 0.304,
        4: 0.380,
        5: 0.438,
        6: 0.482,
        7: 0.517,
    }
    expected_theta_sparsity_value_on_iteration = {
        0: 0.008,
        1: 0.025,
        2: 0.127,
        3: 0.227,
        4: 0.273,
        5: 0.292,
        6: 0.302,
        7: 0.309,
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

        # add perplexity score
        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'PerplexityScore'
        ref_score_config.type = constants.ScoreConfig_Type_Perplexity
        ref_score_config.config = messages.PerplexityScoreConfig().SerializeToString()

        # add sparsity Phi score
        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'SparsityPhiScore'
        ref_score_config.type = constants.ScoreConfig_Type_SparsityPhi
        ref_score_config.config = messages.SparsityPhiScoreConfig().SerializeToString()

        # add sparsity Theta score
        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'SparsityThetaScore'
        ref_score_config.type = constants.ScoreConfig_Type_SparsityTheta
        ref_score_config.config = messages.SparsityThetaScoreConfig().SerializeToString()  

        # add top tokens score
        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'TopTokensScore'
        ref_score_config.type = constants.ScoreConfig_Type_TopTokens
        ref_score_config.config = messages.TopTokensScoreConfig().SerializeToString()

        master_id = lib.ArtmCreateMasterComponent(master_config)

        # Import the collection dictionary
        dict_args = messages.ImportDictionaryArgs()
        dict_args.dictionary_name = 'dictionary'
        dict_args.file_name = os.path.join(batches_folder, dictionary_name)
        lib.ArtmImportDictionary(master_id, dict_args)

        # Configure basic regularizers
        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = 'SmoothSparsePhi'
        ref_reg_config.type = constants.RegularizerConfig_Type_SmoothSparsePhi
        ref_reg_config.config = messages.SmoothSparsePhiConfig().SerializeToString()
        lib.ArtmCreateRegularizer(master_id, ref_reg_config)

        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = 'SmoothSparseTheta'
        ref_reg_config.type = constants.RegularizerConfig_Type_SmoothSparseTheta
        ref_reg_config.config = messages.SmoothSparseThetaConfig().SerializeToString()
        lib.ArtmCreateRegularizer(master_id, ref_reg_config)

        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = 'DecorrelatorPhi'
        ref_reg_config.type = constants.RegularizerConfig_Type_DecorrelatorPhi
        ref_reg_config.config = messages.DecorrelatorPhiConfig().SerializeToString()
        lib.ArtmCreateRegularizer(master_id, ref_reg_config)

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
        proc_args.inner_iterations_count = inner_iterations_count

        # Create configuration for Phi regularization
        reg_args = messages.RegularizeModelArgs()
        reg_args.pwt_source_name = pwt
        reg_args.nwt_source_name = nwt
        reg_args.rwt_target_name = rwt
        
        reg_set = reg_args.regularizer_settings.add()
        reg_set.name = 'SmoothSparsePhi'
        reg_set.tau = smsp_phi_tau
        reg_set.use_relative_regularization = False

        reg_set = reg_args.regularizer_settings.add()
        reg_set.name = 'DecorrelatorPhi'
        reg_set.tau = decor_phi_tau
        reg_set.use_relative_regularization = False

        # Create configuration for Phi normalization
        norm_args = messages.NormalizeModelArgs()
        norm_args.pwt_target_name = pwt
        norm_args.nwt_source_name = nwt
        norm_args.rwt_source_name = rwt

        # Create config for scores retrieval
        perplexity_args = messages.GetScoreValueArgs()
        perplexity_args.model_name = pwt
        perplexity_args.score_name = 'PerplexityScore'

        top_tokens_args = messages.GetScoreValueArgs()
        top_tokens_args.model_name = pwt
        top_tokens_args.score_name = 'TopTokensScore'

        sparsity_phi_args = messages.GetScoreValueArgs()
        sparsity_phi_args.model_name = pwt
        sparsity_phi_args.score_name = 'SparsityPhiScore'

        sparsity_theta_args = messages.GetScoreValueArgs()
        sparsity_theta_args.model_name = pwt
        sparsity_theta_args.score_name = 'SparsityThetaScore'

        for iter in xrange(outer_iterations_count):
            # Invoke one scan of the collection, regularize and normalize Phi
            lib.ArtmRequestProcessBatches(master_id, proc_args)
            lib.ArtmRegularizeModel(master_id, reg_args)
            lib.ArtmNormalizeModel(master_id, norm_args)    

            # Retrieve perplexity score
            results = lib.ArtmRequestScore(master_id, perplexity_args)
            score_data = messages.ScoreData()
            score_data.ParseFromString(results)
            perplexity_score = messages.PerplexityScore()
            perplexity_score.ParseFromString(score_data.data)

            # Retrieve sparsity phi score
            results = lib.ArtmRequestScore(master_id, sparsity_phi_args)
            score_data = messages.ScoreData()
            score_data.ParseFromString(results)
            sparsity_phi_score = messages.SparsityPhiScore()
            sparsity_phi_score.ParseFromString(score_data.data)

            # Retrieve sparsity theta score
            results = lib.ArtmRequestScore(master_id, sparsity_theta_args)
            score_data = messages.ScoreData()
            score_data.ParseFromString(results)
            sparsity_theta_score = messages.SparsityThetaScore()
            sparsity_theta_score.ParseFromString(score_data.data)

            # assert and print scores
            string = 'Iter#{0}'.format(iter)
            string += ': Perplexity = {0:.3f}'.format(perplexity_score.value)
            string += ', Phi sparsity = {0:.3f}'.format(sparsity_phi_score.value)
            string += ', Theta sparsity = {0:.3f}'.format(sparsity_theta_score.value)
            print string

            assert abs(perplexity_score.value - expected_perplexity_value_on_iteration[iter]) < tolerance
            assert abs(sparsity_phi_score.value - expected_phi_sparsity_value_on_iteration[iter]) < tolerance
            assert abs(sparsity_theta_score.value - expected_theta_sparsity_value_on_iteration[iter]) < tolerance

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
                string += ' {0}({1:.3f})'.format(token, weight)
            print string
    finally:
        shutil.rmtree(batches_folder)
