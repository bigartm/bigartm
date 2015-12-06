import os
import numpy
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
    docword = 'docword.kos.txt'
    vocab = 'vocab.kos.txt'
    num_topics = 10
    
    num_tokens = 3501

    batches_folder = tempfile.mkdtemp()
    try:
        # Create the instance of low-level API and master object
        lib = artm.wrapper.LibArtm()
        
        # Parse collection from disk
        lib.ArtmParseCollection({'format': constants.CollectionParserConfig_Format_BagOfWordsUci,
                                 'docword_file_path': os.path.join(os.getcwd(), docword),
                                 'vocab_file_path': os.path.join(os.getcwd(), vocab),
                                 'target_folder': batches_folder})

        # Create master component
        master = mc.MasterComponent(lib)

        # Create collection dictionary and import it
        args = messages.GatherDictionaryArgs()
        args.dictionary_target_name = dictionary_name
        args.data_path = batches_folder
        args.vocab_file_path = os.path.join(os.getcwd(), vocab)
        lib.ArtmGatherDictionary(master.master_id, args)

        # filter the dictionary
        args = messages.FilterDictionaryArgs()
        args.dictionary_name = dictionary_name
        args.dictionary_target_name = dictionary_name + '__'
        args.max_df = 500
        args.min_df = 20
        lib.ArtmFilterDictionary(master.master_id, args)

        # Initialize topic model
        master.initialize_model(model_name=pwt,
                                num_topics=num_topics,
                                disk_path=batches_folder,
                                dictionary_name=dictionary_name + '__')

        # Extract topic model and print extracted data
        info = master.get_phi_info(model=pwt)
        matrix = master.get_phi_matrix(model=pwt)
        print len(info.token) , num_tokens
        assert len(info.token) == num_tokens
        assert numpy.count_nonzero(matrix) == matrix.size
        print 'Number of tokens in Phi matrix = {0}'.format(len(info.token))
        print matrix
    finally:
        shutil.rmtree(batches_folder)
