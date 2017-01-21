# Copyright 2017, Additive Regularization of Topic Models.

from __future__ import print_function

import os
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
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'

    num_topics = 10
    num_document_passes = 1
    num_outer_iterations = 5
    index_to_zero = 4
    zero_tol = 1e-37

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
        scores = {'ThetaSnippet': messages.ThetaSnippetScoreConfig()}
        master = mc.MasterComponent(lib, scores=scores)

        # Create collection dictionary and import it
        master.gather_dictionary(dictionary_target_name=dictionary_name,
                                 data_path=batches_folder,
                                 vocab_file_path=os.path.join(data_path, vocab))

        # Initialize model
        master.initialize_model(model_name=pwt,
                                topic_names=['topic_{}'.format(i) for i in range(num_topics)],
                                dictionary_name=dictionary_name)

        # Attach Pwt matrix
        topic_model, numpy_matrix = master.attach_model(pwt)
        numpy_matrix[:, index_to_zero] = 0

        # Perform iterations
        for iter in range(num_outer_iterations):
            master.clear_score_cache()
            master.process_batches(pwt, nwt, num_document_passes, batches_folder)
            master.normalize_model(pwt, nwt) 

        theta_snippet_score = master.get_score('ThetaSnippet')

        print('ThetaSnippetScore.')
         # Note that 5th topic is fully zero; this is because we performed "numpy_matrix[:, 4] = 0".
        snippet_tuples = zip(theta_snippet_score.values, theta_snippet_score.item_id)
        print_string = ''
        for values, item_id in snippet_tuples:
            print_string += 'Item# {0}:\t'.format(item_id)
            for index, value in enumerate(values.value):
                if index == index_to_zero:
                    assert value < zero_tol
                print_string += '{0:.3f}\t'.format(value)
            print(print_string)
            print_string = ''
    finally:
        shutil.rmtree(batches_folder)
