# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import pytest

from six.moves import range

import artm

def test_func():
    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()
    
    num_tokens = 6906
    num_filtered_tokens = 2852
    num_rate_filtered_tokens = 122
    eps = 1e-5
    
    def _check_num_tokens_in_saved_text_dictionary(file_name, case_type=0):
        with open(file_name, 'r') as fin:
            fin.readline()
            fin.readline()
            counter = 0
            value = 0.0

            for line in fin:
                splitted = line.split(' ')
                if len(splitted) == 5:
                    counter += 1
                    value += float(splitted[2][: -1])

            if case_type == 0:
                assert counter == num_tokens
                assert abs(value - 1.0) < eps
            elif case_type == 1:
                assert counter == num_tokens
                assert abs(value - 0.0) < eps
            elif case_type == 2:
                assert counter == num_filtered_tokens
                assert abs(value - 1.0) < eps
            elif case_type == 3:
                assert counter == num_rate_filtered_tokens
                assert abs(value - 1.0) < eps

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        dictionary_1 = artm.Dictionary()
        dictionary_1.gather(data_path=batches_folder)
        dictionary_1.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_1.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_1.txt'))

        dictionary_1.save(dictionary_path=os.path.join(batches_folder, 'saved_dict_1'))
        dictionary_2 = artm.Dictionary(dictionary_path=os.path.join(batches_folder, 'saved_dict_1.dict'))

        dictionary_2.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_2.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_2.txt'))

        dictionary_3 = artm.Dictionary()
        dictionary_3.load_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_2.txt'))
        dictionary_3.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_3.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_3.txt'))

        dictionary_data = artm.messages.DictionaryData()
        for i in range(num_tokens):
            dictionary_data.token.append('{}_'.format(i))
            dictionary_data.class_id.append('@default_class')
            dictionary_data.token_value.append(0.0)
            dictionary_data.token_df.append(0.0)
            dictionary_data.token_tf.append(1.0)
        f = os.path.join(batches_folder, 'saved_text_dict_3.txt')
        dictionary_data.num_items_in_collection = int(open(f).readline()[: -1].split(' ')[3])

        dictionary_4 = artm.Dictionary()
        dictionary_4.create(dictionary_data=dictionary_data)
        dictionary_4.filter()
        dictionary_4.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_4.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_4.txt'), case_type=1)

        dictionary_5 = artm.Dictionary()
        dictionary_5.create(dictionary_data=dictionary_data)
        dictionary_5.filter(recalculate_value=True)
        dictionary_5.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_5.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_5.txt'))

        dictionary_6 = artm.Dictionary()
        dictionary_6.load(dictionary_path=os.path.join(batches_folder, 'saved_dict_1.dict'))
        dictionary_6.filter(min_df=2, max_df=100, min_tf=1, max_tf=20, recalculate_value=True)
        dictionary_6.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_6.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_6.txt'), case_type=2)

        dictionary_6.filter(min_df_rate=0.001, max_df_rate=0.002, recalculate_value=True)
        dictionary_6.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_6.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_6.txt'), case_type=3)
    finally:
        shutil.rmtree(batches_folder)

