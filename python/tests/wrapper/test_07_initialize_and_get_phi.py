import os
import numpy
import tempfile
import shutil
import pytest

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants
import helpers

def test_func():
    # Set some constants
    dictionary_name = 'dictionary'
    pwt = 'pwt'
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'
    num_topics = 10
    
    num_tokens = 6104

    batches_folder = tempfile.mkdtemp()
    try:
        # Create the instance of low-level API and helper object
        lib = artm.wrapper.LibArtm()
        helper = helpers.TestHelper(lib)
        
        # Parse collection from disk
        lib.ArtmParseCollection({'format': constants.CollectionParserConfig_Format_BagOfWordsUci,
                                 'docword_file_path': os.path.join(os.getcwd(), docword),
                                 'vocab_file_path': os.path.join(os.getcwd(), vocab),
                                 'target_folder': batches_folder,
                                 'dictionary_file_name': dictionary_name})

        # Create master component
        helper.master_id = helper.create_master_component()

        init_args = messages.InitializeModelArgs()
        init_args.model_name = pwt
        init_args.topics_count = num_topics
        init_args.source_type = constants.InitializeModelArgs_SourceType_Batches
        init_args.disk_path = batches_folder
        init_filter = init_args.filter.add()
        init_filter.max_percentage = 0.2  # filter out frequent tokens present in 20% (or more) of items in the collection
        init_filter.min_items = 10  # filter out rare tokens that are present in up to 10 items

        # Use the following option to separately filter each modality.
        # By default filter is applied to all modalities.
        # init_filter.class_id = ... (for example "@default_class")

        # The following alternatives are also available, but they are not as useful as previous options.
        # init_filter.min_percentage = ...    # opposite to max_percentage
        # init_filter.max_items = ...         # opposite to min_items
        # init_filter.min_total_count = 50    # filter out tokens that have in total less than 50 occurrences in collection
        # init_filter.min_one_item_count = 5  # use tokens only if they are present 5 or more times in a single item

        # Initialize topic model
        helper.initialize_model(args=init_args)

        # Extract topic model and print extracted data
        info = helper.get_phi_info(model=pwt)
        matrix = helper.get_phi_matrix(model=pwt)
        assert len(info.token) == num_tokens
        assert numpy.count_nonzero(matrix) == matrix.size
        print 'Number of tokens in Phi matrix = {0}'.format(len(info.token))
        print matrix
    finally:
        shutil.rmtree(batches_folder)
