# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import pytest

from six.moves import range, zip

import artm


def test_func():
    num_topics = 5
    batches_folder = tempfile.mkdtemp()

    try:
        with open(os.path.join(batches_folder, 'temp.vw.txt'), 'w') as fout:
            fout.write('title_0 aaa bbb . aaa ccc  ccc. ccc \n')
            fout.write('title_1 aaa . bbb ccc . ccc bbb .\n')
            fout.write('title_2 aaa bbb ccc\n')
            fout.write('title_3 aaa . bbb . ccc .\n')

        batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(batches_folder, 'temp.vw.txt'),
                                                data_format='vowpal_wabbit',
                                                target_folder=batches_folder)
        model = artm.ARTM(num_topics=num_topics,
                          dictionary=batch_vectorizer.dictionary,
                          num_document_passes=1,
                          cache_theta=True,
                          theta_columns_naming='title')

        model.regularizers.add(artm.TopicSegmentationPtdwRegularizer(
            tau=10.0,
            background_topic_names=['topic_0'],
            merge_into_segments=True,
            threshold=-0.2,
        ))
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        real_values = [
            [
                0.250, 0.114, 0.338, 0.250, 0.052, 0.045, 0.052, 0.271, 0.364,
                0.072, 0.076, 0.364, 0.076, 0.072, 0.364, 0.212, 0.050, 0.044,
                0.288, 0.348, 0.100, 0.348, 0.133, 0.348
            ],
            [
                0.0, 0.0, 0.315, 0.749, 0.947, 0.954, 0.947, 0.0, 0.028, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.028, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.101,
                0.866, 0.101
            ],
            [
                0.749, 0.885, 0.204, 0.0, 0.0, 0.0, 0.0, 0.0, 0.118, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.118, 0.0, 0.0, 0.0, 0.711, 0.651, 0.899, 0.228,
                0.0, 0.228
            ],
            [
                0.0, 0.0, 0.031, 0.0, 0.0, 0.0, 0.0, 0.728, 0.207, 0.927, 0.923,
                0.635, 0.923, 0.927, 0.207, 0.787, 0.949, 0.955, 0.0, 0.0, 0.0,
                0.019, 0.0, 0.019
            ],
            [
                0.0, 0.0, 0.109, 0.0, 0.0, 0.0, 0.0, 0.0, 0.280, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.280, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.301, 0.0,
                0.301
            ]
        ]

        ptdw = model.transform(batch_vectorizer, theta_matrix_type='dense_ptdw')

        for elems, values in zip(ptdw.values.tolist(), real_values):
            for e, v in zip(elems, values):
                assert abs(e - v) < 0.01
    finally:
        shutil.rmtree(batches_folder)
