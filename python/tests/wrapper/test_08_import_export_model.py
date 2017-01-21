# Copyright 2017, Additive Regularization of Topic Models.

from __future__ import print_function

import os
import uuid
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
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'

    num_topics = 10
    num_document_passes = 1
    num_outer_iterations = 2

    batches_folder = tempfile.mkdtemp()
    model_filename = os.path.join(batches_folder, str(uuid.uuid1()))
    try:
        # Create the instance of low-level API and master object
        lib = artm.wrapper.LibArtm()
        
        # Parse collection from disk
        lib.ArtmParseCollection({'format': constants.CollectionParserConfig_CollectionFormat_BagOfWordsUci,
                                 'docword_file_path': os.path.join(data_path, docword),
                                 'vocab_file_path': os.path.join(data_path, vocab),
                                 'target_folder': batches_folder})

        # Create master component
        master = mc.MasterComponent(lib)

        # Create collection dictionary and import it
        master.gather_dictionary(dictionary_target_name=dictionary_name,
                                 data_path=batches_folder,
                                 vocab_file_path=os.path.join(data_path, vocab))

        # Initialize model
        master.initialize_model(model_name=pwt,
                                topic_names=['topic_{}'.format(i) for i in range(num_topics)],
                                dictionary_name=dictionary_name)
        phi_matrix_info = master.get_phi_info(model=pwt)
        
        # Export initialized model
        master.export_model(pwt, model_filename)

        # Create new master component
        master_new = mc.MasterComponent(lib)
        
        # Import model into new master component
        master_new.import_model(pwt, model_filename)
        phi_matrix_info_new = master_new.get_phi_info(model=pwt)
        assert phi_matrix_info.token == phi_matrix_info_new.token
        assert phi_matrix_info_new.num_topics == num_topics
        
        print_string = 'Number of topic in new model is'
        print_string += ' {0} and number of tokens is {1}'.format(phi_matrix_info_new.num_topics,
                                                                  len(phi_matrix_info.token))
        print(print_string)
    finally:
        shutil.rmtree(batches_folder)
