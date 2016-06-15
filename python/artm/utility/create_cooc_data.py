# Utility to generate pairwise coocurancy information of tokens
# of collection in BigARTM batches with default dictionary.
# It will create 'cooc_data.txt' file in the directory with
# this script, that should be passed into artm.Dictionary.gather()
# method as 'cooc_data_path' parameter.

# NOTE: to use script on data with several separated modalities
#       ensure the sections of any two modalities inside one
#       document have no intersections (e.g. each doment has
#       structure [all tokens of class_id_1] [all tokens of class_id_2] ...).
#       This can be reached by using VW file with the same
#       structure during batches creation.

# Author: Murat Apishev (great-mel@yandex.ru)

import os
import sys
import glob
import time
import artm


HELP_STR = '\nUsage: python create_cooc_dictionary'
HELP_STR += '<folder_with_batches_and_dictionary> [<window_size>] [<merge_modalities>]\n'
HELP_STR += '<window_size> is int greater than zero, <merge_modalities> is 0 or 1\n'
HELP_STR += 'if <window_size> is not specified, whole document window will be used\n'
HELP_STR += 'if <merge_modalities> is not specified, tokens will be gathered only within each modality\n\n'

SEPARATOR = '_|@|_'


def __read_params():
    window_size = -1  # window size equal to whole document
    merge_modalities = False
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        raise RuntimeError(HELP_STR)
    if len(sys.argv) > 2:
        window_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        merge_modalities = True if int(sys.argv[3]) == 1 else False

    batches_folder = sys.argv[1]

    return batches_folder, window_size, merge_modalities


def __create_batch_dictionary(batch):
    batch_dictionary = {}
    for index, token, class_id in zip(range(len(batch.token)), batch.token, batch.class_id):
        batch_dictionary[index] = '{0}{1}{2}'.format(token, SEPARATOR, class_id)

    return batch_dictionary


def __save_dictionary(cooc_dictionary, dictionary):
    dict_for_search = {}
    for index, key in enumerate(dictionary):
        dict_for_search[key] = index

    with open('cooc_data.txt', 'w') as fout:
        for index, token_class_id in enumerate(dictionary):
            if token_class_id in cooc_dictionary:
                for key, value in cooc_dictionary[token_class_id].iteritems():
                    fout.write('{0} {1} {2}\n'.format(index, dict_for_search[key], value))


def __process_batch(global_cooc_dictionary, batch, merge_modalities, window_size):
    batch_dictionary = __create_batch_dictionary(batch)

    def __process_window(token_ids, token_weights):
        for i in xrange(len(token_ids)):
            for j in xrange(i + 1, len(token_ids)):
                value = min(token_weights[i], token_weights[j])
                token_1 = batch_dictionary[token_ids[i]]
                token_2 = batch_dictionary[token_ids[j]]
                class_id_1 = token_1.split(SEPARATOR)[1]
                class_id_2 = token_2.split(SEPARATOR)[1]

                if token_1 == token_2:
                    continue

                if token_1 in global_cooc_dictionary:
                    if token_2 in global_cooc_dictionary:
                        if token_2 in global_cooc_dictionary[token_1]:
                            global_cooc_dictionary[token_1][token_2] += value
                        else:
                            if token_1 in global_cooc_dictionary[token_2]:
                                global_cooc_dictionary[token_2][token_1] += value
                            else:
                                if merge_modalities or (class_id_1 == class_id_2):
                                    global_cooc_dictionary[token_1][token_2] = value
                    else:
                        if token_2 in global_cooc_dictionary[token_1]:
                            global_cooc_dictionary[token_1][token_2] += value
                        else:
                            if merge_modalities or (class_id_1 == class_id_2):
                                global_cooc_dictionary[token_1][token_2] = value
                else:
                    if token_2 in global_cooc_dictionary:
                        if token_1 in global_cooc_dictionary[token_2]:
                            global_cooc_dictionary[token_2][token_1] += value
                        else:
                            if merge_modalities or (class_id_1 == class_id_2):
                                global_cooc_dictionary[token_2][token_1] = value
                    else:
                        global_cooc_dictionary[token_1] = {}
                        if merge_modalities or (class_id_1 == class_id_2):
                            global_cooc_dictionary[token_1][token_2] = value

    for item in batch.item:
        real_window_size = window_size if window_size > 0 else len(item.token_id)
        for window_start_id in xrange(0, len(item.token_id) - real_window_size + 1):
            window = {}
            for i, w in zip(item.token_id[window_start_id: (window_start_id + real_window_size)],
                            item.token_weight[window_start_id: (window_start_id + real_window_size)]):
                if not i in window:
                    window[i] = 0
                window[i] += w

            token_ids = sorted(window.keys())
            token_weights = [window[i] for i in token_ids]
            __process_window(token_ids, token_weights)


if __name__ == "__main__":
    global_time_start = time.time()
    batches_folder, window_size, merge_modalities = __read_params()
    batches_list = glob.glob(os.path.join(batches_folder, '*.batch'))
    dictionaries_list = [os.path.join(batches_folder, name) for name in glob.glob(os.path.join(batches_folder, '*.dict'))]

    if len(batches_list) < 1 or len(dictionaries_list) < 1:
        raise RuntimeError('No batches or dictionary were found in given folder')
    else:
        print '{} batches were found, start processing'.format(len(batches_list))

    temp_dict = artm.messages.DictionaryData()
    with open(dictionaries_list[0], 'rb') as fin:
        temp_dict.ParseFromString(fin.read())
    global_dictionary = ['{0}{1}{2}'.format(t, SEPARATOR, c) for t, c in zip(temp_dict.token, temp_dict.class_id)]

    global_cooc_dictionary = {}
    for index, filename in enumerate(batches_list):
        local_time_start = time.time()
        print 'Process batch: {}'.format(index)
        current_batch = artm.messages.Batch()
        with open(filename, 'rb') as fin:
            current_batch.ParseFromString(fin.read())
        __process_batch(global_cooc_dictionary, current_batch, merge_modalities, window_size)

        print 'Finished batch, elapsed time: {}'.format(time.time() - local_time_start)

    __save_dictionary(global_cooc_dictionary, global_dictionary)
    print 'Finished collection, elapsed time: {}'.format(time.time() - global_time_start)
