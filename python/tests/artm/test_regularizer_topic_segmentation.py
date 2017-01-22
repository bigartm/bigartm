# Copyright 2017, Additive Regularization of Topic Models.

from __future__ import print_function
import shutil
import glob
import tempfile
import os
import pytest
import numpy as np

import artm


def test_func():
    # constants
    num_collection_passes = 1
    num_document_passes = 1
    num_topics = 10
    vocab_size = 6906
    num_docs = 3430
    num_tokens = 10
    background_topics = ["topic_0", "topic_1", "topic_2"]
    window = 3
    threshold = 0.2

    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=batch_vectorizer.data_path)

        model_reg = artm.ARTM(num_topics=num_topics, dictionary=dictionary, cache_theta=True, reuse_theta=True)
        model = artm.ARTM(num_topics=num_topics, dictionary=dictionary, cache_theta=True, reuse_theta=True)

        model_reg.num_document_passes = num_document_passes
        model.num_document_passes = num_document_passes

        model_reg.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=num_tokens))
        model.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=num_tokens))

        model_reg.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)
        model_reg.regularizers.add(artm.TopicSegmentationPtdwRegularizer(name='TopicSegmentation', window=window, threshold=threshold, background_topic_names=background_topics))
        ptdw = model.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type='dense_ptdw')
        ptdw_reg = model_reg.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type='dense_ptdw')

        doc_ids = np.random.choice(np.arange(3430), size=500)  #choose some documents randomly
        for doc_id in doc_ids:
            doc_df = ptdw[ptdw.columns[doc_id]]
            token_id = window + np.random.choice(np.arange(doc_df.shape[1] - 2 * window))
            weights = []

            def __f(x):
                temp = 1 - (x[0] + x[1] + x[2])
                weights.append(temp)
                return x * temp

            left_dist = doc_df.iloc[:, token_id - window : token_id].apply(__f, axis=0).sum(axis=1) / sum(weights)
            weights = []
            right_dist = doc_df.iloc[:, token_id : token_id + window].apply(__f, axis=0).sum(axis=1) / sum(weights)

            l = left_dist.argmax()
            r = right_dist.argmax()
            changes_topic = ((left_dist[l] - left_dist[r]) / 2 + (right_dist[r] - right_dist[l]) / 2 > threshold)

            if changes_topic:
                assert r == ptdw_reg[ptdw_reg.columns[doc_id]].iloc[:, token_id].argmax()
            else:
                assert ptdw_reg[ptdw_reg.columns[doc_id]].iloc[:, token_id].argmax() == ptdw_reg[ptdw_reg.columns[doc_id]].iloc[:, token_id - 1].argmax()

    finally:
        shutil.rmtree(batches_folder)
