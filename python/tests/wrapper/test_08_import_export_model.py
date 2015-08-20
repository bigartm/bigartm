import os
import uuid
import tempfile
import shutil
import pytest

import artm.wrapper
import helpers

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
        # Create the instance of low-level API and helper object
        lib = artm.wrapper.LibArtm()
        helper = helpers.TestHelper(lib)
        
        # Parse collection from disk
        helper.parse_collection_uci(os.path.join(os.getcwd(), docword),
                                    os.path.join(os.getcwd(), vocab),
                                    batches_folder,
                                    dictionary_name)

        # Create master component
        helper.master_id = helper.create_master_component()

        # Import the collection dictionary
        helper.import_dictionary(os.path.join(batches_folder, dictionary_name), dictionary_name)

        # Initialize model
        helper.initialize_model(pwt, num_topics, source_type='dictionary', dictionary_name=dictionary_name)
        phi_matrix_info = helper.get_phi_info(model=pwt)
        
        # Export initialized model
        helper.export_model(pwt, model_filename)

        # Create new master component
        master_id_new = helper.create_master_component()
        
        # Import model into new master component
        helper.import_model(pwt, model_filename, master_id=master_id_new)
        phi_matrix_info_new = helper.get_phi_info(model=pwt, master_id=master_id_new)
        assert phi_matrix_info.token == phi_matrix_info_new.token
        assert phi_matrix_info_new.topics_count == num_topics
        
        print_string = 'Number of topic in new model is'
        print_string += ' {0} and number of tokens is {1}'.format(len(phi_matrix_info.token),
                                                                  phi_matrix_info_new.topics_count)
        print print_string
    finally:
        shutil.rmtree(batches_folder)
