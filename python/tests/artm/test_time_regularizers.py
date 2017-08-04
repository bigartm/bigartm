# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import pytest

from six.moves import range, zip

import artm
import pandas as pd


def test_func():
    num_topics = 20
    tolerance = 0.01
    first_sparsity = 0.189
    second_sparsity = 0.251

    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        model = artm.ARTM(num_topics=num_topics, dictionary=batch_vectorizer.dictionary)

        model.scores.add(artm.SparsityPhiScore(name='sp_phi_one', topic_names=model.topic_names[0: 10]))
        model.scores.add(artm.SparsityPhiScore(name='sp_phi_two', topic_names=model.topic_names[10: ]))

        model.regularizers.add(artm.SmoothTimeInTopicsPhiRegularizer(tau=1000.0, topic_names=model.topic_names[0: 10]))

        model.fit_offline(batch_vectorizer, 20)

        assert abs(model.score_tracker['sp_phi_one'].last_value - first_sparsity) < tolerance
        assert abs(model.score_tracker['sp_phi_two'].last_value - second_sparsity) < tolerance
    finally:
        shutil.rmtree(batches_folder)
