import time
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
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'
    num_processors_list = [4, 2, 1]

    num_topics = 10
    num_inner_iterations = 10
    num_outer_iterations = 5

    perplexity_tol = 0.001
    expected_perplexity_value_on_iteration = {
        0: 6699.124,
        1: 2449.832,
        2: 2211.953,
        3: 1938.363,
        4: 1767.228
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

        for num_processors in num_processors_list:
            # Create master component and add scores
            master_config = messages.MasterComponentConfig()
            master_config.processors_count = num_processors

            ref_score_config = master_config.score_config.add()
            ref_score_config.name = 'PerplexityScore'
            ref_score_config.type = constants.ScoreConfig_Type_Perplexity
            ref_score_config.config = messages.PerplexityScoreConfig().SerializeToString()

            master_id = lib.ArtmCreateMasterComponent(master_config)

            # Import the collection dictionary
            dict_args = messages.ImportDictionaryArgs()
            dict_args.dictionary_name = dictionary_name
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

            # Create config for scores retrieval
            perplexity_args = messages.GetScoreValueArgs()
            perplexity_args.model_name = pwt
            perplexity_args.score_name = 'PerplexityScore'

            times = []
            for iter in xrange(num_outer_iterations):
                start = time.time()
                
                # Invoke one scan of the collection and normalize Phi
                lib.ArtmRequestProcessBatches(master_id, proc_args)
                lib.ArtmNormalizeModel(master_id, norm_args)

                # Retrieve and print perplexity score
                results = lib.ArtmRequestScore(master_id, perplexity_args)
                score_data = messages.ScoreData()
                score_data.ParseFromString(results)
                perplexity_score = messages.PerplexityScore()
                perplexity_score.ParseFromString(score_data.data)

                end = time.time()
                assert abs(expected_perplexity_value_on_iteration[iter] - perplexity_score.value) < perplexity_tol
                times.append(end - start)
                string = 'Iter#{0}'.format(iter)
                string += ': Perplexity = {0:.3f}, Time = {1:.3f}'.format(perplexity_score.value, end - start)
                print string

            print 'Average time per iteration = {0:.3f}'.format(float(sum(times)) / len(times))
    finally:
        shutil.rmtree(batches_folder)
