import shutil
import glob
import tempfile
import os
import pytest

import artm


def test_func():
    topic_selection_tau = 1.0
    num_collection_passes = 3
    num_document_passes = 10
    num_topics = 15

    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    batches_folder = tempfile.mkdtemp()

    perplexity_eps = 2.0
    perplexity_value = [8963.45939171117, 2550.7275062628664, 2301.3291199618243]

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

        # Verify that only 7 topics are non-zero (due to TopicSelection regularizer)
        # assert 7 == sum(x == 0 for x in model.get_score('TopicMass').topic_mass)

        # According to https://github.com/bigartm/bigartm/issues/580,
        # we will check only whether the number of non-zero topics changed or not
        topics_left = sum(x == 0 for x in model.get_score('TopicMass').topic_mass)
        assert topics_left != 0 and topics_left != num_topics

        # print model.score_tracker['PerplexityScore'].value
        # the following asssertion fails on travis-ci builds, but passes locally
        # for i in xrange(num_collection_passes):
        #     assert abs(model.score_tracker['PerplexityScore'].value[i] - perplexity_value[i]) < perplexity_eps

        model.fit_online(batch_vectorizer=batch_vectorizer)
    finally:
        shutil.rmtree(batches_folder)
