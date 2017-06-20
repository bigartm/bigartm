# Copyright 2017, Additive Regularization of Topic Models.

from __future__ import print_function

import time
import os
import itertools
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
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'
    num_processors_list = [4, 2, 1]

    num_topics = 10
    num_document_passes = 10
    num_outer_iterations = 5

    perplexity_tol = 1.0
    expected_perplexity_value_on_iteration = {
        0: 6710.208,
        1: 2434.135,
        2: 2202.418,
        3: 1936.493,
        4: 1774.600
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

        for num_processors in num_processors_list:
            # Create master component and scores
            scores = {'PerplexityScore': messages.PerplexityScoreConfig()}
            master = mc.MasterComponent(lib, scores=scores)

            # Create collection dictionary and import it
            master.gather_dictionary(dictionary_target_name=dictionary_name,
                                     data_path=batches_folder,
                                     vocab_file_path=os.path.join(data_path, vocab))

            # Initialize model
            master.initialize_model(model_name=pwt,
                                    topic_names=['topic_{}'.format(i) for i in range(num_topics)],
                                    dictionary_name=dictionary_name)

            times = []
            for iter in range(num_outer_iterations):
                start = time.time()
                
                # Invoke one scan of the collection and normalize Phi
                master.clear_score_cache()
                master.process_batches(pwt, nwt, num_document_passes, batches_folder)
                master.normalize_model(pwt, nwt)  

                # Retrieve and print perplexity score
                perplexity_score = master.get_score('PerplexityScore')

                end = time.time()
                assert abs(expected_perplexity_value_on_iteration[iter] - perplexity_score.value) < perplexity_tol
                times.append(end - start)
                string = 'Iter#{0}'.format(iter)
                string += ': Perplexity = {0:.3f}, Time = {1:.3f}'.format(perplexity_score.value, end - start)
                print(string)

            print('Average time per iteration = {0:.3f}'.format(float(sum(times)) / len(times)))
    finally:
        shutil.rmtree(batches_folder)
