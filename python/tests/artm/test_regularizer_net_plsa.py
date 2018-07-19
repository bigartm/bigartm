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
            fout.write('title_0 aaa:6 bbb:3 ccc:2 |@time_class time_1\n')
            fout.write('title_1 aaa:2 bbb:9 ccc:3\n')
            fout.write('title_2 aaa:1 bbb:2 ccc:7 |@time_class time_2\n')
            fout.write('title_3 aaa:7 bbb:4 ccc:5 |@time_class time_2\n')

        batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(batches_folder, 'temp.vw.txt'),
                                                data_format='vowpal_wabbit',
                                                target_folder=batches_folder)
        # configure model 1
        model = artm.ARTM(num_topics=num_topics,
                          dictionary=batch_vectorizer.dictionary,
                          num_document_passes=1)

        reg = artm.NetPlsaPhiRegularizer(name='net_plsa', tau=1.0, class_id='@time_class',
                                         vertex_names=['time_1', 'time_2'], vertex_weights=[1.0, 2.0],
                                         edge_weights={0: {1: 3.0}, 1: {0: 2.0}})
        model.regularizers.add(reg)

        # configure model 2
        model_2 = artm.ARTM(num_topics=num_topics,
                            dictionary=batch_vectorizer.dictionary,
                            num_document_passes=1)

        model_2.regularizers.add(artm.NetPlsaPhiRegularizer(name='net_plsa', tau=1.0))
        model_2.regularizers['net_plsa'].class_id = '@time_class'
        model_2.regularizers['net_plsa'].vertex_names = ['time_1', 'time_2']
        model_2.regularizers['net_plsa'].vertex_weights = [1.0, 2.0]
        model_2.regularizers['net_plsa'].edge_weights = {0: {1: 3.0}, 1: {0: 2.0}}

        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=2)
        model_2.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=2)

        phi = model.get_phi()
        phi_2 = model_2.get_phi()
        assert phi.equals(phi_2)

        model.dump_artm_model(os.path.join(batches_folder, 'target'))
        model_3 = artm.load_artm_model(os.path.join(batches_folder, 'target'))

        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)
        model_3.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        phi = model.get_phi()
        phi_3 = model_3.get_phi()
        assert phi.equals(phi_3)

        def _f(w):
            return ('@default_class', w)

        def _t(w):
            return ('@time_class', w)

        real_topics = pd.DataFrame(columns=['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4'],
                                   index=[_f('ccc'), _f('bbb'), _f('aaa'), _t('time_1'), _t('time_2')],
                                   data=[[0.098, 0.892, 0.099, 0.389, 0.184],
                                         [0.145, 0.004, 0.618, 0.334, 0.684],
                                         [0.757, 0.104, 0.283, 0.277, 0.132],
                                         [0.06,  0.0,   0.092, 0.0,   0.0  ],
                                         [0.94,  1.0,   0.908, 1.0,   1.0  ]])

        assert (phi - real_topics).abs().values.max() < tolerance 
    finally:
        shutil.rmtree(batches_folder)
