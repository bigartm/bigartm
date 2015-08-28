import os
import uuid
import tempfile
import shutil
import pytest

import artm.wrapper
import artm.wrapper.constants as constants
import artm.master_component as mc

def test_func():
    # Set some constants
    dictionary_name = 'dictionary'
    pwt = 'pwt'
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'

    num_topics = 10
    num_inner_iterations = 1
    num_outer_iterations = 2

    batches_folder = tempfile.mkdtemp()
    model_filename = os.path.join(batches_folder, str(uuid.uuid1()))
    try:
        # Create the instance of low-level API and master object
        lib = artm.wrapper.LibArtm()
        
        # Parse collection from disk
        lib.ArtmParseCollection({'format': constants.CollectionParserConfig_Format_BagOfWordsUci,
                                 'docword_file_path': os.path.join(os.getcwd(), docword),
                                 'vocab_file_path': os.path.join(os.getcwd(), vocab),
                                 'target_folder': batches_folder,
                                 'dictionary_file_name': dictionary_name})

        # Create master component
        master = mc.MasterComponent(lib)

        # Import the collection dictionary
        master.import_dictionary(os.path.join(batches_folder, dictionary_name), dictionary_name)

        # Initialize model
        master.initialize_model(pwt, num_topics, source_type='dictionary', dictionary_name=dictionary_name)
        phi_matrix_info = master.get_phi_info(model=pwt)
        
        # Export initialized model
        master.export_model(pwt, model_filename)

        # Create new master component
        master_new = mc.MasterComponent(lib)
        
        # Import model into new master component
        master_new.import_model(pwt, model_filename)
        phi_matrix_info_new = master_new.get_phi_info(model=pwt)
        assert phi_matrix_info.token == phi_matrix_info_new.token
        assert phi_matrix_info_new.topics_count == num_topics
        
        print_string = 'Number of topic in new model is'
        print_string += ' {0} and number of tokens is {1}'.format(len(phi_matrix_info.token),
                                                                  phi_matrix_info_new.topics_count)
        print print_string
    finally:
        shutil.rmtree(batches_folder)
