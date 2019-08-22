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
    num_topics = 5
    tolerance = 0.01
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
                          num_document_passes=1)

        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DPR', tau=1))
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        def _f(w):
            return ('@default_class', w)

        phi = model.get_phi()
        real_topics = pd.DataFrame(columns=['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4'],
                                   index=[_f('ccc'), _f('bbb'), _f('aaa')],
                                   data=[[0.32, 0.95, 0.2,  0.55, 0.32],
                                         [0.33, 0.0,  0.68, 0.35, 0.63],
                                         [0.35, 0.05, 0.12, 0.1,  0.05]])

        assert (phi - real_topics).abs().values.max() < tolerance

        model.regularizers['DPR'].topic_names = [model.topic_names[0], model.topic_names[1]]
        model.regularizers['DPR'].topic_pairs = {model.topic_names[0]: {model.topic_names[1]: 100.0,
                                                                        model.topic_names[2]: 100.0}}
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        phi = model.get_phi()
        real_topics = pd.DataFrame(columns=['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4'],
                                   index=[_f('ccc'), _f('bbb'), _f('aaa')],
                                   data=[[0.0, 0.94, 0.22, 0.58, 0.35],
                                         [0.0, 0.0,  0.63, 0.3,  0.58],
                                         [0.0, 0.06, 0.15, 0.12, 0.07]])

        assert (phi - real_topics).abs().values.max() < tolerance

        model.regularizers['DPR'].topic_pairs = {model.topic_names[1]: {model.topic_names[0]: 10000.0}}
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        phi = model.get_phi()
        real_topics = pd.DataFrame(columns=['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4'],
                                   index=[_f('ccc'), _f('bbb'), _f('aaa')],
                                   data=[[0.0, 0.91, 0.21, 0.54, 0.35],
                                         [0.0, 0.0,  0.55, 0.26, 0.53],
                                         [0.0, 0.09, 0.24, 0.20, 0.12]])

        assert (phi - real_topics).abs().values.max() < tolerance
    finally:
        shutil.rmtree(batches_folder)
