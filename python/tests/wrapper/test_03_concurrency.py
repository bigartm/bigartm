import time
import os
import itertools
import tempfile
import shutil
import pytest

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants
import artm.master_component as mc

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
        lib.ArtmParseCollection({'format': constants.CollectionParserConfig_Format_BagOfWordsUci,
                                 'docword_file_path': os.path.join(os.getcwd(), docword),
                                 'vocab_file_path': os.path.join(os.getcwd(), vocab),
                                 'target_folder': batches_folder,
                                 'dictionary_file_name': dictionary_name})

        for num_processors in num_processors_list:
            # Create master component and scores
            scores = [('PerplexityScore', messages.PerplexityScoreConfig())]
            master = mc.MasterComponent(lib, scores=scores)

            # Import the collection dictionary
            master.import_dictionary(os.path.join(batches_folder, dictionary_name), dictionary_name)

            # Initialize model
            master.initialize_model(pwt, num_topics, source_type='dictionary', dictionary_name=dictionary_name)

            times = []
            for iter in xrange(num_outer_iterations):
                start = time.time()
                
                # Invoke one scan of the collection and normalize Phi
                master.process_batches(pwt, nwt, num_inner_iterations, batches_folder, reset_scores=True)
                master.normalize_model(pwt, nwt)  

                # Retrieve and print perplexity score
                perplexity_score = master.retrieve_score(pwt, 'PerplexityScore')

                end = time.time()
                assert abs(expected_perplexity_value_on_iteration[iter] - perplexity_score.value) < perplexity_tol
                times.append(end - start)
                string = 'Iter#{0}'.format(iter)
                string += ': Perplexity = {0:.3f}, Time = {1:.3f}'.format(perplexity_score.value, end - start)
                print string

            print 'Average time per iteration = {0:.3f}'.format(float(sum(times)) / len(times))
    finally:
        shutil.rmtree(batches_folder)
