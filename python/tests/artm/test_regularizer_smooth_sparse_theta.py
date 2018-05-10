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
            fout.write('title_0 aaa:1 bbb:2 ccc:3\n')
            fout.write('title_1 aaa:1 bbb:2 ccc:3\n')
            fout.write('title_2 aaa:1 bbb:2 ccc:3\n')
            fout.write('title_3 aaa:1 bbb:2 ccc:3\n')

        batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(batches_folder, 'temp.vw.txt'),
                                                data_format='vowpal_wabbit',
                                                target_folder=batches_folder)
        model = artm.ARTM(num_topics=num_topics,
                          dictionary=batch_vectorizer.dictionary,
                          num_document_passes=1,
                          cache_theta=True,
                          theta_columns_naming='title')

        model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SST',
                                                                 tau=-1000.0,
                                                                 doc_titles=['title_0', 'title_2']))
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        theta = model.get_theta()
        real_values = [
            [0.0, 0.14, 0.0, 0.14],
            [0.0, 0.25, 0.0, 0.25],
            [0.0, 0.19, 0.0, 0.19],
            [0.0, 0.21, 0.0, 0.21],
            [0.0, 0.21, 0.0, 0.21],
        ]

        for elems, values in zip(theta.values.tolist(), real_values):
            for e, v in zip(elems, values):
                assert abs(e - v) < 0.01

        model.initialize(dictionary=batch_vectorizer.dictionary)
        model.regularizers['SST'].doc_titles=['title_0', 'title_2', 'title_1']
        model.regularizers['SST'].doc_topic_coef=[0.0, 1.0, 1.0, 0.0, 0.0]
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        theta = model.get_theta()
        real_values = [
            [0.26, 0.26, 0.26, 0.14],
            [ 0.0,  0.0,  0.0, 0.25],
            [ 0.0,  0.0,  0.0, 0.19],
            [0.36, 0.36, 0.36, 0.21],
            [0.38, 0.38, 0.38, 0.21],
        ]

        for elems, values in zip(theta.values.tolist(), real_values):
            for e, v in zip(elems, values):
                assert abs(e - v) < 0.01

        model.initialize(dictionary=batch_vectorizer.dictionary)
        model.regularizers['SST'].doc_titles=['title_0', 'title_3']
        model.regularizers['SST'].doc_topic_coef=[[-1.0, 1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0, 0.0]]
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        theta = model.get_theta()
        real_values = [
            [0.499311, 0.146202, 0.146202, 0.000873],
            [     0.0, 0.247351, 0.247351,      0.0],
            [0.000556, 0.185883, 0.185883, 0.001110],
            [0.000617, 0.206015, 0.206015, 0.996735],
            [0.499516, 0.214550, 0.214550, 0.001282],
        ]

        for elems, values in zip(theta.values.tolist(), real_values):
            for e, v in zip(elems, values):
                assert abs(e - v) < 0.000001

        model.initialize(dictionary=batch_vectorizer.dictionary)
        model.regularizers['SST'].doc_titles=[]
        model.regularizers['SST'].doc_topic_coef=[0.0, 1.0, 1.0, 0.0, 0.0]
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        theta = model.get_theta()
        real_values = [
            [0.26, 0.26, 0.26, 0.26],
            [ 0.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0],
            [0.36, 0.36, 0.36, 0.36],
            [0.38, 0.38, 0.38, 0.38],
        ]

        for elems, values in zip(theta.values.tolist(), real_values):
            for e, v in zip(elems, values):
                assert abs(e - v) < 0.01
    finally:
        shutil.rmtree(batches_folder)
