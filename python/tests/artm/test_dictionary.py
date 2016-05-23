import shutil
import glob
import tempfile
import os
import pytest

import artm

def test_func():
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    batches_folder = tempfile.mkdtemp()
    
    num_tokens = 6906
    num_filtered_tokens = 2852
    
    def _check_num_tokens_in_saved_text_dictionary(file_name, filtered=False):
        with open(file_name, 'r') as fin:
            fin.next()
            fin.next()
            counter = 0
            for line in fin:
                counter += 1
            assert counter == (num_tokens if not filtered else num_filtered_tokens)

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
        for i in xrange(num_tokens):
            dictionary_data.token.append('{}_'.format(i))
            dictionary_data.class_id.append('@default_class')
            dictionary_data.token_value.append(0.0)
            dictionary_data.token_df.append(0.0)
            dictionary_data.token_tf.append(0.0)

        dictionary_4 = artm.Dictionary()
        dictionary_4.create(dictionary_data=dictionary_data)
        dictionary_4.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_4.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_4.txt'))

        dictionary_5 = artm.Dictionary()
        dictionary_5.load(dictionary_path=os.path.join(batches_folder, 'saved_dict_1.dict'))
        dictionary_5.filter(min_df=2, max_df=100, min_tf=1, max_tf=20)
        dictionary_5.save_text(dictionary_path=os.path.join(batches_folder, 'saved_text_dict_5.txt'))
        _check_num_tokens_in_saved_text_dictionary(os.path.join(batches_folder, 'saved_text_dict_5.txt'), filtered=True)
    finally:
        shutil.rmtree(batches_folder)
