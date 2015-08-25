# -*- coding: utf-8 -*-

import os
import uuid
import string
import itertools
import tempfile
import shutil
import pytest

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.master_component as mc

def _print_top_tokens(top_tokens_score, expected_values_topic, tolerance):
    top_tokens_triplets = zip(top_tokens_score.topic_index,
                              zip(top_tokens_score.token,
                                  top_tokens_score.weight))
    for topic_index, group in itertools.groupby(top_tokens_triplets, key=lambda (topic_index, _): topic_index):
        print_string = u'Topic#{0} : '.format(topic_index)
        for _, (token, weight) in group:
            print_string += u' {0}({1:.3f})'.format(token, weight)
            assert abs(expected_values_topic[topic_index][token] - weight) < tolerance
        print print_string

def test_func():
    # Set some constants
    dictionary_name = 'dictionary'
    pwt = 'pwt'
    nwt = 'nwt'

    num_topics = 2
    num_inner_iterations = 10
    num_outer_iterations = 10
    
    russian_class_weight = 1.0
    english_class_weight = 1.0
    russian_class = '@russian'
    english_class = '@english'

    tolerance = 0.001
    expected_values_rus_topic = {
        0: {
            u'документ': 0.125,
            u'текст': 0.125,
            u'анализ': 0.125,
            u'статистический': 0.125,
            u'модель': 0.125,
            u'коллекция': 0.083,
            u'тематическая': 0.083,
            'model': 0.042,
            'topic': 0.042,
            'artm': 0.042
        },
        1: {
            u'ногие': 0.115,
            u'отряд': 0.115,
            u'млекопитающие': 0.115,
            u'семейство': 0.115,
            u'хищный': 0.077,
            u'ласто': 0.077,
            u'моржовых': 0.077,
            u'тюлень': 0.077,
            u'ушастый': 0.077,
            u'коротко': 0.038
        }
    }
    expected_values_eng_topic = {
        0: {
            'model': 0.167,
            'text': 0.125,
            'analysis': 0.125,
            'statistical': 0.125,
            'topic': 0.125,
            'artm': 0.083,
            'plsa': 0.083,
            'lda': 0.083,
            'collection': 0.083,
            'not': 0.000
        },
        1: {
            'mammal': 0.188,
            'predatory': 0.125,
            'eared': 0.125,
            'marine': 0.125,
            'seal': 0.125,
            'not': 0.062,
            'reptile': 0.062,
            'crocodilia': 0.062,
            'order': 0.062,
            'pinnipeds': 0.062
        }
    }
    expected_sparsity_values = {'russian': 0.5, 'english': 0.5}

    # Prepare multimodal data
    ens = []
    rus = []

    ens.append(u'Topic model statistical analysis text collection LDA PLSA ARTM')
    rus.append(u'Тематическая модель статистический анализ текст коллекция')

    ens.append(u'LDA statistical topic model text collection')
    rus.append(u'LDA статистический тематическая модель текст документ коллекция')

    ens.append(u'PLSA statistical analysis text model')
    rus.append(u'PLSA статистический анализ документ текст модель')

    ens.append(u'ARTM analysis topic model')
    rus.append(u'ARTM анализ документ topic model')

    ens.append(u'Pinnipeds seal marine mammal order')
    rus.append(u'Тюлень семейство млекопитающие моржовых отряд ласто ногие')

    ens.append(u'Eared seal marine predatory mammal')
    rus.append(u'Ушастый тюлень семейство млекопитающие отряд хищный семейство моржовых ласто ногие')

    ens.append(u'Eared Crocodilia predatory reptile not mammal')
    rus.append(u'Ушастый крокодил гена отряд хищный не млекопитающие коротко ногие')

    ru_dic = {}  # mapping from russian token to its index in batch.token list
    en_dic = {}  # mapping from english token to its index in batch.token list
    batch = messages.Batch()  # batch representing the entire collection
    batch.id = str(uuid.uuid1())
    dict_config = messages.DictionaryConfig()  # BigARTM dictionary to initialize model

    def append(tokens, dic, item, class_id):
        for token in tokens:
            if not dic.has_key(token):              # New token discovered:
                dic[token] = len(batch.token)       # 1. update ru_dic or en_dic
                batch.token.append(token)           # 2. update batch.token and batch.class_id
                batch.class_id.append(class_id)
                entry = dict_config.entry.add()   # 3. update dict_config
                entry.key_token = token
                entry.class_id = class_id

            # Add token to the item.
            item.field[0].token_id.append(dic[token])
            # replace '1' with the actual number of token occupancies in the item
            item.field[0].token_count.append(1)

    # Iterate through all items and populate the batch
    for (en, ru) in zip(ens, rus):
        next_item = batch.item.add()
        next_item.id = len(batch.item) - 1
        next_item.field.add()
        append(string.split(ru.lower()), ru_dic, next_item, russian_class)
        append(string.split(en.lower()), en_dic, next_item, english_class)

    batches_folder = tempfile.mkdtemp()
    try:
        # Create the instance of low-level API and master object
        lib = artm.wrapper.LibArtm()

        # Save batch and dictionary on the disk
        lib.ArtmSaveBatch(batches_folder, batch)

        dict_config.name = dictionary_name
        with open(os.path.join(batches_folder, dictionary_name), 'wb') as file:
            file.write(dict_config.SerializeToString())

        # Create master component and scores
        scores = [('SparsityPhiRus', messages.SparsityPhiScoreConfig(class_id = russian_class)),
                  ('SparsityPhiEng', messages.SparsityPhiScoreConfig(class_id = english_class)),
                  ('TopTokensRus', messages.TopTokensScoreConfig(class_id=russian_class)),
                  ('TopTokensEng', messages.TopTokensScoreConfig(class_id = english_class))]
        master = mc.MasterComponent(lib, scores=scores)

        # Import the collection dictionary
        master.import_dictionary(os.path.join(batches_folder, dictionary_name), dictionary_name)

        # Initialize model
        master.initialize_model(pwt, num_topics, source_type='dictionary', dictionary_name=dictionary_name)

        for iter in xrange(num_outer_iterations):
            # Invoke one scan of the collection, regularize and normalize Phi
            master.process_batches(pwt, nwt, num_inner_iterations, batches_folder,
                                   class_ids=[russian_class, english_class],
                                   class_weights=[russian_class_weight, english_class_weight],
                                   reset_scores=True)
            master.normalize_model(pwt, nwt)    

        # Retrieve and print scores
        top_tokens_rus = master.retrieve_score(pwt, 'TopTokensRus')
        top_tokens_eng = master.retrieve_score(pwt, 'TopTokensEng')
        sp_phi_rus = master.retrieve_score(pwt, 'SparsityPhiRus')
        sp_phi_eng = master.retrieve_score(pwt, 'SparsityPhiEng')

        print 'Top tokens per russian topic:'
        _print_top_tokens(top_tokens_rus, expected_values_rus_topic, tolerance)
        print 'Top tokens per english topic:'
        _print_top_tokens(top_tokens_eng, expected_values_eng_topic, tolerance)

        print '\nSparsity Phi: russian {0:.3f}, english {1:.3f}'.format(sp_phi_rus.value, sp_phi_eng.value)
        assert abs(expected_sparsity_values['russian'] - sp_phi_rus.value) < tolerance
        assert abs(expected_sparsity_values['english'] - sp_phi_eng.value) < tolerance
    finally:
        shutil.rmtree(batches_folder)
