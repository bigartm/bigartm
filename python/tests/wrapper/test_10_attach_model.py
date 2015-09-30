import os
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

    num_topics = 10
    num_inner_iterations = 1
    num_outer_iterations = 5
    index_to_zero = 4
    zero_tol = 1e-37

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

        # Create master component and scores
        scores = [('ThetaSnippet', messages.ThetaSnippetScoreConfig())]
        master = mc.MasterComponent(lib, scores=scores)

        # Initialize model
        master.initialize_model(pwt, num_topics, source_type='batches', disk_path=batches_folder)

        # Attach Pwt matrix
        topic_model, numpy_matrix = master.attach_model(pwt)
        numpy_matrix[:, index_to_zero] = 0

        # Perform iterations
        for iter in xrange(num_outer_iterations):
            master.process_batches(pwt, nwt, num_inner_iterations, batches_folder, reset_scores=True)
            master.normalize_model(pwt, nwt) 

        theta_snippet_score = master.retrieve_score(pwt, 'ThetaSnippet')

        print 'Option 1. ThetaSnippetScore.'
         # Note that 5th topic is fully zero; this is because we performed "numpy_matrix[:, 4] = 0".
        snippet_tuples = zip(theta_snippet_score.values, theta_snippet_score.item_id)
        print_string = ''
        for values, item_id in snippet_tuples:
            print_string += 'Item# {0}:\t'.format(item_id)
            for index, value in enumerate(values.value):
                if index == index_to_zero:
                    assert value < zero_tol
                print_string += '{0:.3f}\t'.format(value)
            print print_string
            print_string = ''
    finally:
        shutil.rmtree(batches_folder)
