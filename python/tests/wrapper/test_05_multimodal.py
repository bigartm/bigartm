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
import artm.wrapper.constants as constants

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

    tolerance = 0.01
    expected_values_rus_topic = {
        0: {
            u'ногие': 0.115,
            u'отряд': 0.115,
            u'млекопитающие': 0.115,
            u'семейство': 0.115,
            u'хищный': 0.077,
            u'ушастый': 0.077,
            u'ласто': 0.077,
            u'моржовых': 0.077,
            u'тюлень': 0.077,
            u'коротко': 0.038
        },
        1: {
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
        }
    }
    expected_values_eng_topic = {
        0: {
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
        },
        1: {
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
        # Create the instance of low-level API
        lib = artm.wrapper.LibArtm()

        # Save batch and dictionary on the disk
        lib.ArtmSaveBatch(batches_folder, batch)

        dict_config.name = dictionary_name
        with open(os.path.join(batches_folder, dictionary_name), 'wb') as file:
            file.write(dict_config.SerializeToString())

        # Create master component and add scores
        master_config = messages.MasterComponentConfig()

        # Add sparsity Phi scores for russian and english
        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'SparsityPhiRussianScore'
        ref_score_config.type = constants.ScoreConfig_Type_SparsityPhi
        sparsity_score = messages.SparsityPhiScoreConfig()
        sparsity_score.class_id = russian_class
        ref_score_config.config = sparsity_score.SerializeToString()

        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'SparsityPhiEnglishScore'
        ref_score_config.type = constants.ScoreConfig_Type_SparsityPhi
        sparsity_score = messages.SparsityPhiScoreConfig()
        sparsity_score.class_id = english_class
        ref_score_config.config = sparsity_score.SerializeToString()

        # Add top tokens scores for russian and english
        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'TopTokensRussianScore'
        ref_score_config.type = constants.ScoreConfig_Type_TopTokens
        top_tokens_score = messages.TopTokensScoreConfig()
        top_tokens_score.class_id = russian_class
        ref_score_config.config = top_tokens_score.SerializeToString()

        ref_score_config = master_config.score_config.add()
        ref_score_config.name = 'TopTokensEnglishScore'
        ref_score_config.type = constants.ScoreConfig_Type_TopTokens
        top_tokens_score = messages.TopTokensScoreConfig()
        top_tokens_score.class_id = english_class
        ref_score_config.config = top_tokens_score.SerializeToString()

        master_id = lib.ArtmCreateMasterComponent(master_config)        

        # Import the collection dictionary
        dict_args = messages.ImportDictionaryArgs()
        dict_args.dictionary_name = 'dictionary'
        dict_args.file_name = os.path.join(batches_folder, dictionary_name)
        lib.ArtmImportDictionary(master_id, dict_args)

        # Initialize model
        init_args = messages.InitializeModelArgs()
        init_args.model_name = pwt
        init_args.dictionary_name = dictionary_name
        init_args.source_type = constants.InitializeModelArgs_SourceType_Dictionary
        init_args.topics_count = num_topics
        lib.ArtmInitializeModel(master_id, init_args)

        # Create configuration for batch processing
        proc_args = messages.ProcessBatchesArgs()
        proc_args.pwt_source_name = pwt
        proc_args.nwt_target_name = nwt
        
        proc_args.class_id.append(russian_class)
        proc_args.class_id.append(english_class)
        proc_args.class_weight.append(russian_class_weight)
        proc_args.class_weight.append(english_class_weight)
        
        for name in os.listdir(batches_folder):
            if name != dictionary_name:
                proc_args.batch_filename.append(os.path.join(batches_folder, name))
        proc_args.inner_iterations_count = num_inner_iterations

        # Create configuration for Phi normalization
        norm_args = messages.NormalizeModelArgs()
        norm_args.pwt_target_name = pwt
        norm_args.nwt_source_name = nwt

        # Create config for scores retrieval
        sp_phi_rus_args = messages.GetScoreValueArgs()
        sp_phi_rus_args.model_name = pwt
        sp_phi_rus_args.score_name = 'SparsityPhiRussianScore'

        sp_phi_eng_args = messages.GetScoreValueArgs()
        sp_phi_eng_args.model_name = pwt
        sp_phi_eng_args.score_name = 'SparsityPhiEnglishScore'

        top_tokens_rus_args = messages.GetScoreValueArgs()
        top_tokens_rus_args.model_name = pwt
        top_tokens_rus_args.score_name = 'TopTokensRussianScore'

        top_tokens_eng_args = messages.GetScoreValueArgs()
        top_tokens_eng_args.model_name = pwt
        top_tokens_eng_args.score_name = 'TopTokensEnglishScore'

        for iter in xrange(num_outer_iterations):
            # Invoke one scan of the collection, regularize and normalize Phi
            lib.ArtmRequestProcessBatches(master_id, proc_args)
            lib.ArtmNormalizeModel(master_id, norm_args)    

        # Retrieve scores
        results = lib.ArtmRequestScore(master_id, sp_phi_rus_args)
        score_data = messages.ScoreData()
        score_data.ParseFromString(results)
        sp_phi_rus_score = messages.SparsityPhiScore()
        sp_phi_rus_score.ParseFromString(score_data.data)

        results = lib.ArtmRequestScore(master_id, sp_phi_eng_args)
        score_data = messages.ScoreData()
        score_data.ParseFromString(results)
        sp_phi_eng_score = messages.SparsityPhiScore()
        sp_phi_eng_score.ParseFromString(score_data.data)

        results = lib.ArtmRequestScore(master_id, top_tokens_rus_args)
        score_data = messages.ScoreData()
        score_data.ParseFromString(results)
        top_tokens_rus_score = messages.TopTokensScore()
        top_tokens_rus_score.ParseFromString(score_data.data)

        results = lib.ArtmRequestScore(master_id, top_tokens_eng_args)
        score_data = messages.ScoreData()
        score_data.ParseFromString(results)
        top_tokens_eng_score = messages.TopTokensScore()
        top_tokens_eng_score.ParseFromString(score_data.data)

        print 'Top tokens per russian topic:'
        top_tokens_triplets = zip(top_tokens_rus_score.topic_index,
                                  zip(top_tokens_rus_score.token,
                                      top_tokens_rus_score.weight))
        for topic_index, group in itertools.groupby(top_tokens_triplets, key=lambda (topic_index, _): topic_index):
            print_string = u'Topic#{0} : '.format(topic_index)
            for _, (token, weight) in group:
                print_string += u' {0}({1:.3f})'.format(token, weight)
                assert abs(expected_values_rus_topic[topic_index][token] - weight) < tolerance
            print print_string

        print 'Top tokens per english topic:'
        top_tokens_triplets = zip(top_tokens_eng_score.topic_index,
                                  zip(top_tokens_eng_score.token,
                                      top_tokens_eng_score.weight))
        for topic_index, group in itertools.groupby(top_tokens_triplets, key=lambda (topic_index, _): topic_index):
            print_string = u'Topic#{0} : '.format(topic_index)
            for _, (token, weight) in group:
                print_string += u' {0}({1:.3f})'.format(token, weight)
                assert abs(expected_values_eng_topic[topic_index][token] - weight) < tolerance
            print print_string

        print '\nSparsity Phi: russian {0:.3f}, english {1:.3f}'.format(sp_phi_rus_score.value, sp_phi_eng_score.value)
        assert abs(expected_sparsity_values['russian'] - sp_phi_rus_score.value) < tolerance
        assert abs(expected_sparsity_values['english'] - sp_phi_eng_score.value) < tolerance
    finally:
        shutil.rmtree(batches_folder)
