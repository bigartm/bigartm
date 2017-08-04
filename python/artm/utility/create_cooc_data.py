# Copyright 2017, Additive Regularization of Topic Models.

# Utility to generate pairwise co-occurrence information of tokens
# of collection in BigARTM batches with default dictionary.
# It will create 'cooc_data.txt' file in the directory with
# this script, that should be passed into artm.Dictionary.gather()
# method as 'cooc_data_path' parameter.

# Author: Murat Apishev (great-mel@yandex.ru)

from __future__ import print_function

import os
import sys
import glob
import time
import artm
import codecs

from six import iteritems
from six.moves import range

HELP_STR = '\nUsage: python create_cooc_dictionary'
HELP_STR += '<folder_with_batches_and_dictionary> [<window_size>] [<merge_modalities>]\n'
HELP_STR += '<window_size> is int greater than zero, <merge_modalities> is 0 or 1\n'
HELP_STR += 'if <window_size> is not specified, whole document window will be used\n\n'

def __read_params():
    window_size = -1  # window size equal to whole document
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise RuntimeError(HELP_STR)
    if len(sys.argv) > 2:
        window_size = int(sys.argv[2])

    batches_folder = sys.argv[1]

    return batches_folder, window_size


def __create_batch_dictionary(batch):
    batch_dictionary = {}
    for index, token in enumerate(batch.token):
        batch_dictionary[index] = token

    return batch_dictionary


def __save_dictionary(cooc_dictionary, num_tokens):
    with open('cooc_data.txt', 'w') as fout:
        for index in range(num_tokens):
            if index in cooc_dictionary:
                for key, value in iteritems(cooc_dictionary[index]):
                    fout.write(u'{0} {1} {2}\n'.format(index, key, value))


def __process_batch(global_cooc_dictionary, batch, window_size, dictionary):
    batch_dictionary = __create_batch_dictionary(batch)

    def __process_window(token_ids, token_weights):
        for j in range(1, len(token_ids)):
            value = min(token_weights[0], token_weights[j])
            token_index_1 = dictionary[batch_dictionary[token_ids[0]]]
            token_index_2 = dictionary[batch_dictionary[token_ids[j]]]

            if token_index_1 in global_cooc_dictionary:
                if token_index_2 in global_cooc_dictionary:
                    if token_index_2 in global_cooc_dictionary[token_index_1]:
                        global_cooc_dictionary[token_index_1][token_index_2] += value
                    else:
                        if token_index_1 in global_cooc_dictionary[token_index_2]:
                            global_cooc_dictionary[token_index_2][token_index_1] += value
                        else:
                            global_cooc_dictionary[token_index_1][token_index_2] = value
                else:
                    if token_index_2 in global_cooc_dictionary[token_index_1]:
                        global_cooc_dictionary[token_index_1][token_index_2] += value
                    else:
                        global_cooc_dictionary[token_index_1][token_index_2] = value
            else:
                if token_index_2 in global_cooc_dictionary:
                    if token_index_1 in global_cooc_dictionary[token_index_2]:
                        global_cooc_dictionary[token_index_2][token_index_1] += value
                    else:
                        global_cooc_dictionary[token_index_2][token_index_1] = value
                else:
                    global_cooc_dictionary[token_index_1] = {}
                    global_cooc_dictionary[token_index_1][token_index_2] = value

    for item in batch.item:
        real_window_size = window_size if window_size > 0 else len(item.token_id)
        for window_start_id in range(len(item.token_id)):
            end_index = window_start_id + real_window_size
            token_ids = item.token_id[window_start_id: end_index if end_index < len(item.token_id) else len(item.token_id)]
            token_weights = item.token_weight[window_start_id: end_index if end_index < len(item.token_id) else len(item.token_id)]
            __process_window(token_ids, token_weights)

def __size(obj):
    result = sys.getsizeof(global_cooc_dictionary)
    for k_1, internal in iteritems(global_cooc_dictionary):
        result += sys.getsizeof(k_1)
        for t, v in iteritems(internal):
            result += sys.getsizeof(t)
            result += sys.getsizeof(v)

    return result

if __name__ == "__main__":
    global_time_start = time.time()
    batches_folder, window_size = __read_params()
    batches_list = glob.glob(os.path.join(batches_folder, '*.batch'))
    dictionaries_list = [name for name in glob.glob(os.path.join(batches_folder, '*.dict'))]

    if len(batches_list) < 1 or len(dictionaries_list) < 1:
        raise RuntimeError('No batches or dictionaries were found in given folder')
    else:
        print('{} batches were found, start processing'.format(len(batches_list)))

    temp_dict = artm.Dictionary()
    temp_dict.load(dictionaries_list[0])
    file_name = '../cooc_info/{}_temp_dict.txt'.format(time.time())
    temp_dict.save_text(file_name)

    dictionary = {}
    with codecs.open(file_name, 'r', 'utf-8') as fin:
        fin.next()
        fin.next()
        for index, line in enumerate(fin):
            dictionary[line.split(' ')[0][0: -1]] = index
    os.remove(file_name)

    global_cooc_dictionary = {}
    for index, filename in enumerate(batches_list):
        local_time_start = time.time()
        print('Process batch: {}'.format(index))
        current_batch = artm.messages.Batch()
        with open(filename, 'rb') as fin:
            current_batch.ParseFromString(fin.read())
        __process_batch(global_cooc_dictionary, current_batch, window_size, dictionary)

        print('Finished batch, elapsed time: {}'.format(time.time() - local_time_start))

    __save_dictionary(global_cooc_dictionary, len(dictionary.keys()))
    print('Finished collection, elapsed time: {0}, size: {1} Gb'.format(time.time() - global_time_start,
                                                                        __size(global_cooc_dictionary) / 1000000000.0))
