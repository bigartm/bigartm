# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import pytest

from six.moves import range

import artm

def test_func():
    # constants
    num_tokens = 11
    probability_mass_threshold = 0.9
    sp_reg_tau = -0.1
    decor_tau = 1.5e+5
    decor_rel_tau = 0.3
    num_collection_passes = 15
    num_document_passes = 1
    num_topics = 15
    vocab_size = 6906
    num_docs = 3430

    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()

    sp_zero_eps = 0.001
    sparsity_phi_value = [0.034, 0.064, 0.093, 0.120, 0.145,
                          0.170, 0.194, 0.220, 0.246, 0.277,
                          0.312, 0.351, 0.390, 0.428, 0.464]

    sparsity_phi_rel_value = [0.442, 0.444, 0.444, 0.446, 0.448,
                              0.449, 0.458, 0.468, 0.476, 0.488,
                              0.501, 0.522, 0.574, 0.609, 0.670]

    sparsity_theta_value = [0.0] * num_collection_passes

    perp_zero_eps = 2.0
    perplexity_value = [6873, 2590, 2685, 2578, 2603,
                        2552, 2536, 2481, 2419, 2331,
                        2235, 2140, 2065, 2009, 1964]

    perplexity_rel_value = [6873, 2667, 2458, 2323, 2150,
                            2265, 2015, 1967, 1807, 1747,
                            1713, 1607, 1632, 1542, 1469]

    top_zero_eps = 0.0001
    top_tokens_num_tokens = [num_tokens * num_topics] * num_collection_passes
    top_tokens_topic_0_tokens = [u'party', u'state', u'campaign', u'tax',
                                 u'political',u'republican', u'senate', u'candidate',
                                 u'democratic', u'court', u'president']
    top_tokens_topic_0_weights = [0.0209, 0.0104, 0.0094, 0.0084,
                                  0.0068, 0.0067, 0.0065, 0.0058,
                                  0.0053, 0.0053, 0.0051]

    ker_zero_eps = 0.02
    topic_kernel_topic_0_contrast = 0.96
    topic_kernel_topic_0_purity = 0.014
    topic_kernel_topic_0_size = 18.0
    topic_kernel_average_size = [0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.13, 0.6, 1.6,
                                 3.53, 7.15, 12.6, 20.4, 29.06]
    topic_kernel_average_contrast = [0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.12, 0.31, 0.7,
                                     0.96, 0.96, 0.96, 0.96, 0.97]
    topic_kernel_average_purity = [0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.01, 0.01, 0.015,
                                   0.017, 0.02, 0.03, 0.04, 0.05]

    len_last_document_ids = 10

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=batch_vectorizer.data_path)

        model = artm.ARTM(topic_names=['topic_{}'.format(i) for i in range(num_topics)],
                          dictionary=dictionary.name,
                          cache_theta=True)

        model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=sp_reg_tau))
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=decor_tau))

        model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
        model.scores.add(artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary))
        model.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
        model.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=num_tokens))
        model.scores.add(artm.TopicKernelScore(name='TopicKernelScore',
                                               probability_mass_threshold=probability_mass_threshold))
        model.scores.add(artm.ThetaSnippetScore(name='ThetaSnippetScore'))

        model.num_document_passes = num_document_passes
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)

        for i in range(num_collection_passes):
            assert abs(model.score_tracker['SparsityPhiScore'].value[i] - sparsity_phi_value[i]) < sp_zero_eps

        for i in range(num_collection_passes):
            assert abs(model.score_tracker['SparsityThetaScore'].value[i] - sparsity_theta_value[i]) < sp_zero_eps

        for i in range(num_collection_passes):
            assert abs(model.score_tracker['PerplexityScore'].value[i] - perplexity_value[i]) < perp_zero_eps

        for i in range(num_collection_passes):
            assert model.score_tracker['TopTokensScore'].num_tokens[i] == top_tokens_num_tokens[i]

        for i in range(num_tokens):
            assert model.score_tracker['TopTokensScore'].last_tokens[model.topic_names[0]][i] == top_tokens_topic_0_tokens[i]
            assert abs(model.score_tracker['TopTokensScore'].last_weights[model.topic_names[0]][i] - top_tokens_topic_0_weights[i]) < top_zero_eps

        assert len(model.score_tracker['TopicKernelScore'].last_tokens[model.topic_names[0]]) > 0

        assert abs(topic_kernel_topic_0_contrast - model.score_tracker['TopicKernelScore'].last_contrast[model.topic_names[0]]) < ker_zero_eps
        assert abs(topic_kernel_topic_0_purity - model.score_tracker['TopicKernelScore'].last_purity[model.topic_names[0]]) < ker_zero_eps
        assert abs(topic_kernel_topic_0_size - model.score_tracker['TopicKernelScore'].last_size[model.topic_names[0]]) < ker_zero_eps

        for i in range(num_collection_passes):
            assert abs(model.score_tracker['TopicKernelScore'].average_size[i] - topic_kernel_average_size[i]) < ker_zero_eps
            assert abs(model.score_tracker['TopicKernelScore'].average_contrast[i] - topic_kernel_average_contrast[i]) < ker_zero_eps
            assert abs(model.score_tracker['TopicKernelScore'].average_purity[i] - topic_kernel_average_purity[i]) < ker_zero_eps

        model.fit_online(batch_vectorizer=batch_vectorizer)

        info = model.info
        assert info is not None
        assert len(info.config.topic_name) == num_topics
        assert len(info.score) >= len(model.score_tracker)
        assert len(info.regularizer) == len(model.regularizers.data)
        assert len(info.cache_entry) > 0

        temp = model.score_tracker['ThetaSnippetScore'].last_document_ids
        assert len_last_document_ids == len(temp)
        assert len(model.score_tracker['ThetaSnippetScore'].last_snippet[temp[0]]) == num_topics

        phi = model.get_phi()
        assert phi.shape == (vocab_size, num_topics)
        theta = model.get_theta()
        assert theta.shape == (num_topics, num_docs)

        assert model.library_version.count('.') == 2  # major.minor.patch

        # test relative coefficients for Phi matrix regularizers
        model = artm.ARTM(num_topics=num_topics,
                          dictionary=dictionary.name,
                          cache_theta=False)

        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=decor_rel_tau))
        model.regularizers['DecorrelatorPhi'].gamma = 0.0

        model.scores.add(artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary))
        model.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))

        model.num_document_passes = num_document_passes
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)

        model_clone = model.clone()
        assert model_clone is not None

        for m in [model, model_clone]:
            for i in range(num_collection_passes):
                assert abs(m.score_tracker['SparsityPhiScore'].value[i] - sparsity_phi_rel_value[i]) < sp_zero_eps

            for i in range(num_collection_passes):
                assert abs(m.score_tracker['PerplexityScore'].value[i] - perplexity_rel_value[i]) < perp_zero_eps
    finally:
        shutil.rmtree(batches_folder)
