import shutil
import glob
import tempfile
import os
import pytest

import artm

def test_func():
    # constants
    dictionary_name = 'dictionary'
    num_tokens = 11
    probability_mass_threshold = 0.9
    sp_reg_tau = -0.1
    decor_tau = 1.5e+5
    num_collection_passes = 15
    num_document_passes = 1
    num_topics = 15

    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    batches_folder = tempfile.mkdtemp()

    sp_zero_eps = 0.001
    sparsity_phi_value = [0.034, 0.064, 0.093, 0.120, 0.145,
                          0.170, 0.194, 0.220, 0.246, 0.277,
                          0.312, 0.351, 0.390, 0.428, 0.464]
    sparsity_phi_zero_tokens = [3541, 6704, 9648, 12449, 15103,
                                17669, 20171, 22812, 25553, 28747,
                                32418, 36407, 40423, 44414, 48098]
    sparsity_phi_total_tokens = [103590] * num_collection_passes

    sparsity_theta_value = [0.0] * num_collection_passes
    sparsity_theta_zero_topics = [0] * num_collection_passes
    sparsity_theta_total_topics = [51450] * num_collection_passes

    perp_zero_eps = 2.0
    perplexity_value = [6873, 2590, 2685, 2578, 2603,
                        2552, 2536, 2481, 2419, 2331,
                        2235, 2140, 2065, 2009, 1964]
    perplexity_raw = [-4132474, -3676020, -3692813, -3673925, -3678402,
                      -3669127, -3666114, -3655902, -3644084, -3626837,
                      -3607101, -3586853, -3570221, -3557158, -3546595]
    perplexity_normalizer = [467714] * num_collection_passes
    perplexity_zero_tokens = [0] * num_collection_passes

    top_zero_eps= 0.0001
    top_tokens_num_tokens = [num_tokens * num_topics] * num_collection_passes
    top_tokens_average_coherence = [0.0] * num_collection_passes
    top_tokens_topic_0_tokens = [u'party', u'state', u'campaign', u'tax',
                                 u'political',u'republican', u'senate', u'candidate',
                                 u'democratic', u'court', u'president']
    top_tokens_topic_0_weights = [0.0209, 0.0104, 0.0094, 0.0084,
                                  0.0068, 0.0067, 0.0065, 0.0058,
                                  0.0053, 0.0053, 0.0051]

    ker_zero_eps = 0.01
    topic_kernel_topic_0_tokens = dict.fromkeys([u'ceo', u'millionaire', u'income', u'catholics', u'catholic',
                                                 u'bishops', u'cat', u'mouse', u'devos', u'pope',
                                                 u'fatherinlaw', u'michigans', u'charismatic', u'enron', u'blackwell',
                                                 u'walmart', u'communion', u'schwarz'])
    topic_kernel_topic_0_contrast = 0.96
    topic_kernel_topic_0_purity = 0.014
    topic_kernel_topic_0_size = 18.0
    topic_kernel_topic_0_coherence = 0.0
    topic_kernel_average_size = [0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.13, 0.53, 1.6,
                                 3.33, 7.13, 12.067, 19.53, 27.8]
    topic_kernel_average_contrast = [0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.12, 0.25, 0.7,
                                     0.96, 0.96, 0.96, 0.96, 0.97]
    topic_kernel_average_purity = [0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.01, 0.01, 0.015,
                                   0.017, 0.02, 0.03, 0.04, 0.05]
    topic_kernel_average_coherence = [0.0] * num_collection_passes

    snip_zero_eps = 0.0001
    last_document_ids = [3430, 3421, 3422, 3423, 3424,
                         3425, 3426, 3427, 3428, 3429]
    last_snippet_doc_0 = [0.0898, 0.0710, 0.0079, 0.0959, 0.0684,
                          0.0550, 0.0611, 0.0659, 0.0751, 0.0497,
                          0.0198, 0.1128, 0.1098, 0.0558, 0.0612]

    try:
        data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        batch_vectorizer = None
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        model = artm.ARTM(topic_names=['topic_{}'.format(i) for i in xrange(num_topics)])

        model.gather_dictionary(dictionary_name, batch_vectorizer.data_path)
        model.initialize(dictionary_name=dictionary_name)

        model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=sp_reg_tau))
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=decor_tau))

        model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
        model.scores.add(artm.PerplexityScore(name='PerplexityScore',
                                              use_unigram_document_model=False,
                                              dictionary_name=dictionary_name))
        model.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
        model.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=num_tokens))
        model.scores.add(artm.TopicKernelScore(name='TopicKernelScore',
                                               probability_mass_threshold=probability_mass_threshold))
        model.scores.add(artm.ThetaSnippetScore(name='ThetaSnippetScore'))

        model.num_document_passes = num_document_passes
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)

        for i in xrange(num_collection_passes):
            assert abs(model.score_tracker['SparsityPhiScore'].value[i] - sparsity_phi_value[i]) < sp_zero_eps
            assert abs(model.score_tracker['SparsityPhiScore'].zero_tokens[i] - sparsity_phi_zero_tokens[i]) < sp_zero_eps
            assert abs(model.score_tracker['SparsityPhiScore'].total_tokens[i] - sparsity_phi_total_tokens[i]) < sp_zero_eps

        for i in xrange(num_collection_passes):
            assert abs(model.score_tracker['SparsityThetaScore'].value[i] - sparsity_theta_value[i]) < sp_zero_eps
            assert abs(model.score_tracker['SparsityThetaScore'].zero_topics[i] - sparsity_theta_zero_topics[i]) < sp_zero_eps
            assert abs(model.score_tracker['SparsityThetaScore'].total_topics[i] - sparsity_theta_total_topics[i]) < sp_zero_eps

        for i in xrange(num_collection_passes):
            assert abs(model.score_tracker['PerplexityScore'].value[i] - perplexity_value[i]) < perp_zero_eps
            assert abs(model.score_tracker['PerplexityScore'].raw[i] - perplexity_raw[i]) < perp_zero_eps
            assert abs(model.score_tracker['PerplexityScore'].normalizer[i] - perplexity_normalizer[i]) < perp_zero_eps
            assert abs(model.score_tracker['PerplexityScore'].zero_tokens[i] - perplexity_zero_tokens[i]) < perp_zero_eps
            assert abs(model.score_tracker['PerplexityScore'].theta_sparsity_value[i] - sparsity_theta_value[i]) < perp_zero_eps
            assert abs(model.score_tracker['PerplexityScore'].theta_sparsity_zero_topics[i] - sparsity_theta_zero_topics[i]) < perp_zero_eps
            assert abs(model.score_tracker['PerplexityScore'].theta_sparsity_total_topics[i] - sparsity_theta_total_topics[i]) < perp_zero_eps

        for i in xrange(num_collection_passes):
            assert model.score_tracker['TopTokensScore'].num_tokens[i] == top_tokens_num_tokens[i]
            assert model.score_tracker['TopTokensScore'].average_coherence[i] == top_tokens_average_coherence[i]

        for i in xrange(num_tokens):
            assert model.score_tracker['TopTokensScore'].last_tokens[model.topic_names[0]][i] == top_tokens_topic_0_tokens[i]
            assert abs(model.score_tracker['TopTokensScore'].last_weights[model.topic_names[0]][i] - top_tokens_topic_0_weights[i]) < top_zero_eps

        temp = model.score_tracker['TopicKernelScore'].last_tokens[model.topic_names[0]]
        assert len(topic_kernel_topic_0_tokens.keys()) == len(temp)
        for i in xrange(len(temp)):
            assert temp[i] in topic_kernel_topic_0_tokens

        assert abs(topic_kernel_topic_0_contrast - model.score_tracker['TopicKernelScore'].last_contrast[model.topic_names[0]]) < ker_zero_eps
        assert abs(topic_kernel_topic_0_purity - model.score_tracker['TopicKernelScore'].last_purity[model.topic_names[0]]) < ker_zero_eps
        assert abs(topic_kernel_topic_0_size - model.score_tracker['TopicKernelScore'].last_size[model.topic_names[0]]) < ker_zero_eps
        assert abs(topic_kernel_topic_0_coherence - model.score_tracker['TopicKernelScore'].last_coherence[model.topic_names[0]]) < ker_zero_eps

        for i in xrange(num_collection_passes):
            assert abs(model.score_tracker['TopicKernelScore'].average_size[i] - topic_kernel_average_size[i]) < ker_zero_eps
            assert abs(model.score_tracker['TopicKernelScore'].average_contrast[i] - topic_kernel_average_contrast[i]) < ker_zero_eps
            assert abs(model.score_tracker['TopicKernelScore'].average_purity[i] - topic_kernel_average_purity[i]) < ker_zero_eps
            assert abs(model.score_tracker['TopicKernelScore'].average_coherence[i] - topic_kernel_average_coherence[i]) < ker_zero_eps

        temp = model.score_tracker['ThetaSnippetScore'].last_document_ids
        assert len(last_document_ids) == len(temp)
        for i in xrange(len(temp)):
            assert temp[i] == last_document_ids[i]

        temp = model.score_tracker['ThetaSnippetScore'].last_snippet[temp[0]]
        assert len(temp) == num_topics
        for i in xrange(num_topics):
            assert abs(temp[i] - last_snippet_doc_0[i]) < snip_zero_eps
    finally:
        shutil.rmtree(batches_folder)
