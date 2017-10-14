# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import pytest

from six.moves import range

import artm


def test_func():
    topic_selection_tau = 0.5
    num_collection_passes = 3
    num_document_passes = 10
    num_topics = 15

    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()

    perplexity_eps = 0.1
    perplexity_value = [6676.941798754971, 2534.963709464024, 2463.1544861984794]

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        dictionary = artm.Dictionary(data_path=batches_folder)
        model = artm.ARTM(num_topics=num_topics, dictionary=dictionary, num_document_passes=num_document_passes)

        model.regularizers.add(artm.TopicSelectionThetaRegularizer(name='TopicSelection', tau=topic_selection_tau))
        model.scores.add(artm.PerplexityScore(name='PerplexityScore'))
        model.scores.add(artm.TopicMassPhiScore(name='TopicMass', model_name=model.model_nwt))
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)

        # Verify that only 8 topics are non-zero (due to TopicSelection regularizer)
        topics_left = sum(x == 0 for x in model.get_score('TopicMass').topic_mass)
        assert 8 == topics_left

        # the following asssertion fails on travis-ci builds, but passes locally
        for i in range(num_collection_passes):
            assert abs(model.score_tracker['PerplexityScore'].value[i] - perplexity_value[i]) < perplexity_eps

    finally:
        shutil.rmtree(batches_folder)
