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
    num_tokens = 15
    alpha = 0.01
    beta = 0.02
    num_collection_passes = 15
    num_document_passes = 1
    num_topics = 15
    vocab_size = 6906
    num_docs = 3430
    zero_eps = 0.001

    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=batch_vectorizer.data_path)

        model_artm = artm.ARTM(num_topics=num_topics, dictionary=dictionary, cache_theta=True, reuse_theta=True)

        model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=beta))
        model_artm.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=alpha))

        model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
        model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary))
        model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
        model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=num_tokens))

        model_lda = artm.LDA(num_topics=num_topics, alpha=alpha, beta=beta, dictionary=dictionary, cache_theta=True)
        model_lda.initialize(dictionary=dictionary)
 
        model_artm.num_document_passes = num_document_passes
        model_lda.num_document_passes = num_document_passes
        
        model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)
        model_lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)

        for i in range(num_collection_passes):
            assert abs(model_artm.score_tracker['SparsityPhiScore'].value[i] - model_lda.sparsity_phi_value[i]) < zero_eps

        for i in range(num_collection_passes):
            assert abs(model_artm.score_tracker['SparsityThetaScore'].value[i] - model_lda.sparsity_theta_value[i]) < zero_eps

        for i in range(num_collection_passes):
            assert abs(model_artm.score_tracker['PerplexityScore'].value[i] - model_lda.perplexity_value[i]) < zero_eps

        lda_tt = model_lda.get_top_tokens(num_tokens=num_tokens)
        assert len(lda_tt) == num_topics

        for i in range(num_topics):
            for j in range(num_tokens):
                assert model_artm.score_tracker['TopTokensScore'].last_tokens[model_artm.topic_names[i]][j] == lda_tt[i][j]

        lda_tt = model_lda.get_top_tokens(num_tokens=num_tokens, with_weights=True)
        for i in range(num_tokens):
            assert abs(model_artm.score_tracker['TopTokensScore'].last_weights[model_artm.topic_names[0]][i] - lda_tt[0][i][1]) < zero_eps

        model_lda.fit_online(batch_vectorizer=batch_vectorizer)

        phi = model_lda.phi_
        assert phi.shape == (vocab_size, num_topics)
        theta = model_lda.get_theta()
        assert theta.shape == (num_topics, num_docs)

        assert model_lda.library_version.count('.') == 2  # major.minor.patch

        assert model_lda.clone() is not None

        model_lda = artm.LDA(num_topics=num_topics, alpha=alpha, beta=([0.1] * num_topics), dictionary=dictionary, cache_theta=True)
        assert model_lda._internal_model.regularizers.size() == num_topics + 1
    finally:
        shutil.rmtree(batches_folder)
