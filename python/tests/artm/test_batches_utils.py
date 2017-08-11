# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import numpy
import pytest

from six.moves import range

import artm

def test_func():
    # constatnts
    num_uci_batches = 4
    n_wd = numpy.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])
    vocab = {0: 'test', 1: 'artm', 2: 'python', 3: 'batch'}
    num_n_wd_batches = 3
    n_wd_num_tokens = n_wd.shape[0]
    dictionary_name = 'dict.txt'
    n_wd_tokens_list = ['test', 'python', 'artm', 'batch']
    n_wd_token_tf_list = ['15.0', '25.0', '20.0', '30.0']
    n_wd_token_df_list = [str(float(n_wd.shape[1])) + '\n'] * n_wd.shape[0]

    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()
    try:
        uci_batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                    data_format='bow_uci',
                                                    collection_name='kos',
                                                    target_folder=batches_folder)

        assert len(glob.glob(os.path.join(batches_folder, '*.batch'))) == num_uci_batches
        assert len(uci_batch_vectorizer.batches_list) == num_uci_batches


        dictionary = uci_batch_vectorizer.dictionary
        model = artm.ARTM(num_topics=10, dictionary=dictionary)
        model.scores.add(artm.PerplexityScore(name='perplexity', dictionary=dictionary))

        batches = []
        for b in uci_batch_vectorizer.batches_ids:
            batch = artm.messages.Batch()
            with open(b, 'rb') as fin:
                batch.ParseFromString(fin.read())
                batches.append(batch)

        in_memory_batch_vectorizer = artm.BatchVectorizer(data_format='batches',
                                                          process_in_memory_model=model,
                                                          batches=batches)

        model.fit_offline(num_collection_passes=10, batch_vectorizer=in_memory_batch_vectorizer)
        model.fit_online(update_every=1, batch_vectorizer=in_memory_batch_vectorizer)
        assert len(model.score_tracker['perplexity'].value) == 14

        del in_memory_batch_vectorizer


        batch_batch_vectorizer = artm.BatchVectorizer(data_path=batches_folder, data_format='batches')
        assert len(batch_batch_vectorizer.batches_list) == num_uci_batches


        uci_batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                    data_format='bow_uci',
                                                    collection_name='kos')

        temp_target_folder = uci_batch_vectorizer._target_folder
        assert os.path.isdir(temp_target_folder)
        assert len(glob.glob(os.path.join(temp_target_folder, '*.batch'))) == num_uci_batches

        uci_batch_vectorizer.__del__()
        assert not os.path.isdir(temp_target_folder)


        n_wd_batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                     data_format='bow_n_wd',
                                                     n_wd=n_wd,
                                                     vocabulary=vocab,
                                                     batch_size=2)

        temp_target_folder = n_wd_batch_vectorizer._target_folder
        assert os.path.isdir(temp_target_folder)
        assert len(n_wd_batch_vectorizer.batches_list) == num_n_wd_batches
        assert len(glob.glob(os.path.join(temp_target_folder, '*.batch'))) == num_n_wd_batches

        for i in range(num_n_wd_batches):
            with open(n_wd_batch_vectorizer.batches_ids[i], 'rb') as fin:
                batch = artm.messages.Batch()
                batch.ParseFromString(fin.read())
                assert len(batch.item) == 2 or len(batch.item) == 1
                assert len(batch.token) == n_wd_num_tokens
            
        n_wd_batch_vectorizer.dictionary.save_text(os.path.join(temp_target_folder, dictionary_name))
        assert os.path.isfile(os.path.join(temp_target_folder, dictionary_name))
        with open(os.path.join(temp_target_folder, dictionary_name), 'r') as fin:
            counter = 0
            tokens, token_tf, token_df = [], [], []
            for line in fin:
                counter += 1
                if counter > 2:
                    temp = line.split(', ')
                    tokens.append(temp[0])
                    token_tf.append(temp[3])
                    token_df.append(temp[4])
                    
            assert counter == n_wd_num_tokens + 2

            # ToDo: we're not able to compare lists directly in Python 3 because of
            #       unknown reasons. This should be fixed
            assert set(tokens) == set(n_wd_tokens_list)
            assert set(token_tf) == set(n_wd_token_tf_list)
            assert set(token_df) == set(n_wd_token_df_list)

        n_wd_batch_vectorizer.__del__()
        assert not os.path.isdir(temp_target_folder)
    finally:
        shutil.rmtree(batches_folder)
