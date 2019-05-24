# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import pytest
import numpy
import json

from six.moves import range, zip
from six import iteritems

import artm

def _assert_json_params(params):
    params['num_processors'] == 7
    len(params['topic_names']) == 15
    params['model_pwt'] == 'pwt'
    params['num_document_passes'] == 5
    params['synchronizations_processed'] == 14
    params['num_online_processed_batches'] == 4
    params['show_progress_bars'] == False
    params['model_nwt'] == 'nwt'
    len(params['scores']) == 5
    len(params['regularizers']) == 4
    params['transaction_typenames'] == {u'@default_class': 1.0}
    params['class_ids'] == {u'@default_transaction': 1.0}


def _assert_params_equality(model_1, model_2):
    assert model_1.num_processors == model_2.num_processors
    assert model_1.cache_theta == model_2.cache_theta
    assert model_1.num_document_passes == model_2.num_document_passes
    assert model_1.reuse_theta == model_2.reuse_theta
    assert model_1.theta_columns_naming == model_2.theta_columns_naming
    assert model_1.seed == model_2.seed
    assert model_1.show_progress_bars == model_2.show_progress_bars
    assert model_1.topic_names == model_2.topic_names
    assert model_1.class_ids == model_2.class_ids
    assert model_1.transaction_typenames == model_2.transaction_typenames
    assert model_1.model_pwt == model_2.model_pwt
    assert model_1.model_nwt == model_2.model_nwt
    assert model_1.theta_name == model_2.theta_name
    assert model_1._synchronizations_processed == model_2._synchronizations_processed
    assert model_1._num_online_processed_batches == model_2._num_online_processed_batches
    assert model_1._initialized == model_2._initialized


def _assert_scores_equality(model_1, model_2):
    assert set(model_1.scores.data.keys()) == set(model_2.scores.data.keys())

    for name in model_1.scores.data.keys():
        if name == 'perp':
            assert model_1.scores[name].dictionary == model_2.scores[name].dictionary
        elif name == 'sp_theta':
            assert abs(model_1.scores[name].eps - model_2.scores[name].eps) < 1e-5
        elif name == 'top_tok':
            assert model_1.scores[name].num_tokens == model_2.scores[name].num_tokens
            assert model_1.scores[name].class_id == model_2.scores[name].class_id
        elif name == 'sp_nwt':
            assert model_1.scores[name].model_name == model_2.scores[name].model_name
        elif name == 'kernel':
            assert model_1.scores[name].topic_names == model_2.scores[name].topic_names
            assert abs(model_1.scores[name].probability_mass_threshold -
                       model_2.scores[name].probability_mass_threshold) < 1e-5
        else:
            raise RuntimeError('No such score: {}'.format(name))

def _assert_regularizers_equality(model_1, model_2):
    assert set(model_1.regularizers.data.keys()) == set(model_2.regularizers.data.keys())

    for name in model_1.regularizers.data.keys():
        assert model_1.regularizers[name].tau == model_2.regularizers[name].tau
        if name == 'decor':
            assert model_1.regularizers[name].gamma == model_2.regularizers[name].gamma
            assert model_1.regularizers[name].topic_pairs == model_2.regularizers[name].topic_pairs
        elif name == 'smsp_phi':
            assert model_1.regularizers[name].gamma == model_2.regularizers[name].gamma
            assert model_1.regularizers[name].dictionary == model_2.regularizers[name].dictionary
        elif name == 'smsp_theta':
            assert model_1.regularizers[name].doc_topic_coef == model_2.regularizers[name].doc_topic_coef
        elif name == 'sm_ptdw':
            pass
        else:
            raise RuntimeError('No such regularizers: {}'.format(name))


def _assert_score_values_equality(model_1, model_2):
    assert set(model_1.scores.data.keys()) == set(model_2.scores.data.keys())

    for name in model_1.scores.data.keys():
        if name == 'perp' or name == 'sp_theta' or name == 'sp_nwt':
            assert sum([abs(x - y) for x, y in zip(model_1.score_tracker[name].value,
                                                   model_2.score_tracker[name].value)]) < 0.005
        elif name == 'top_tok':
            assert set(model_1.score_tracker[name].last_tokens) == set(model_2.score_tracker[name].last_tokens)
        elif name == 'kernel':
            assert set(model_1.score_tracker[name].last_tokens) == set(model_2.score_tracker[name].last_tokens)
            assert sum([abs(x - y) < 1e-8 for x, y in zip(model_1.score_tracker[name].average_size,
                                                          model_2.score_tracker[name].average_size)])
            assert sum([abs(x - y) < 1e-8 for x, y in zip(model_1.score_tracker[name].average_contrast,
                                                          model_2.score_tracker[name].average_contrast)])
            assert sum([abs(x - y) < 1e-8 for x, y in zip(model_1.score_tracker[name].average_purity,
                                                          model_2.score_tracker[name].average_purity)])
            assert len(model_1.score_tracker[name].contrast)  == len(model_2.score_tracker[name].contrast)
        else:
            raise RuntimeError('No such score: {}'.format(name))


def _assert_matrices_equality(model_1, model_2):
    assert numpy.allclose(model_1.get_phi().as_matrix(), model_2.get_phi().as_matrix(), 0.0001, 0.0001)
    assert numpy.allclose(model_1.get_phi(model_name=model_1.model_nwt).as_matrix(),
                          model_2.get_phi(model_name=model_1.model_nwt).as_matrix(), 0.0001, 0.0001)
    if model_1.theta_name is not None:
        assert numpy.allclose(model_1.get_phi(model_name=model_1.theta_name).as_matrix(),
                              model_2.get_phi(model_name=model_1.theta_name).as_matrix(), 0.0001, 0.0001)


def test_func():
    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()
    dump_folder = tempfile.mkdtemp()

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        model_1 = artm.ARTM(num_processors=7,
                            cache_theta=True,
                            num_document_passes=5,
                            reuse_theta=True,
                            seed=10,
                            num_topics=15,
                            class_ids={'@default_class': 1.0},
                            theta_name='THETA',
                            dictionary=batch_vectorizer.dictionary)

        model_2 = artm.ARTM(num_processors=7,
                            cache_theta=False,
                            num_document_passes=5,
                            reuse_theta=False,
                            seed=10,
                            num_topics=15,
                            class_ids={'@default_class': 1.0},
                            transaction_typenames={'@default_transaction': 1.0},
                            dictionary=batch_vectorizer.dictionary)

        for model in [model_1, model_2]:
            model.scores.add(artm.PerplexityScore(name='perp', dictionary=batch_vectorizer.dictionary))
            model.scores.add(artm.SparsityThetaScore(name='sp_theta', eps=0.1))
            model.scores.add(artm.TopTokensScore(name='top_tok', num_tokens=10))
            model.scores.add(artm.SparsityPhiScore(name='sp_nwt', model_name=model.model_nwt))
            model.scores.add(artm.TopicKernelScore(name='kernel', topic_names=model.topic_names[0: 5],
                                                   probability_mass_threshold=0.4))

            topic_pairs = {}
            for topic_name_1 in model.topic_names:
                for topic_name_2 in model.topic_names:
                    if topic_name_1 not in topic_pairs:
                        topic_pairs[topic_name_1] = {}
                    topic_pairs[topic_name_1][topic_name_2] = numpy.random.randint(0, 3)

            model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decor', tau=100000.0,
                                                                   topic_pairs=topic_pairs))
            model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='smsp_phi', tau=-0.5, gamma=0.3,
                                                                   dictionary=batch_vectorizer.dictionary))
            model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='smsp_theta', tau=0.1,
                                                                     doc_topic_coef=[2.0] * model.num_topics))
            model.regularizers.add(artm.SmoothPtdwRegularizer(name='sm_ptdw', tau=0.1))

            # learn first model and dump it on disc
            model.fit_offline(batch_vectorizer, num_collection_passes=10)
            model.fit_online(batch_vectorizer, update_every=1)

            model.dump_artm_model(os.path.join(dump_folder, 'target'))

            params = {}
            with open(os.path.join(dump_folder, 'target', 'parameters.json'), 'r') as fin:
                params = json.load(fin)
            _assert_json_params(params)

            # create second model from the dump and check the results are equal
            model_new = artm.load_artm_model(os.path.join(dump_folder, 'target'))

            _assert_params_equality(model, model_new)
            _assert_scores_equality(model, model_new)
            _assert_regularizers_equality(model, model_new)
            _assert_score_values_equality(model, model_new)
            _assert_matrices_equality(model, model_new)
         
            # continue learning of both models
            model.fit_offline(batch_vectorizer, num_collection_passes=3)
            model.fit_online(batch_vectorizer, update_every=1)

            model_new.fit_offline(batch_vectorizer, num_collection_passes=3)
            model_new.fit_online(batch_vectorizer, update_every=1)

            # check new results are also equal
            _assert_params_equality(model, model_new)
            _assert_scores_equality(model, model_new)
            _assert_regularizers_equality(model, model_new)
            _assert_score_values_equality(model, model_new)
            _assert_matrices_equality(model, model_new)

            shutil.rmtree(os.path.join(dump_folder, 'target'))
    finally:
        shutil.rmtree(batches_folder)
        shutil.rmtree(dump_folder)
