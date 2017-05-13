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

        phi = model.get_phi()
        real_topics = pd.DataFrame(
            data={
                'topic_0': dict(ccc=0.32, bbb=0.33, aaa=0.35),
                'topic_1': dict(ccc=0.95, bbb=0.0, aaa=0.05),
                'topic_2': dict(ccc=0.2, bbb=0.68, aaa=0.12),
                'topic_3': dict(ccc=0.55, bbb=0.35, aaa=0.1),
                'topic_4': dict(ccc=0.32, bbb=0.63, aaa=0.05),
            }
        )
        assert (phi - real_topics).abs().values.max() < tolerance

        model.regularizers['DPR'].topic_names = [model.topic_names[0], model.topic_names[1]]
        model.regularizers['DPR'].topic_pairs = {model.topic_names[0]: {model.topic_names[1]: 100.0,
                                                                        model.topic_names[2]: 100.0}}
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        phi = model.get_phi()
        real_topics = pd.DataFrame(
            data={
                'topic_0': dict(ccc=0.0, bbb=0.0, aaa=0.0),
                'topic_1': dict(ccc=0.94, bbb=0.0, aaa=0.06),
                'topic_2': dict(ccc=0.22, bbb=0.63, aaa=0.15),
                'topic_3': dict(ccc=0.58, bbb=0.3, aaa=0.12),
                'topic_4': dict(ccc=0.35, bbb=0.58, aaa=0.07),
            }
        )
        assert (phi - real_topics).abs().values.max() < tolerance

        model.regularizers['DPR'].topic_pairs = {model.topic_names[1]: {model.topic_names[0]: 10000.0}}
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

        phi = model.get_phi()
        real_topics = pd.DataFrame(
            data={
                'topic_0': dict(ccc=0.0, bbb=0.0, aaa=0.0),
                'topic_1': dict(ccc=0.91, bbb=0.0, aaa=0.09),
                'topic_2': dict(ccc=0.21, bbb=0.55, aaa=0.24),
                'topic_3': dict(ccc=0.54, bbb=0.26, aaa=0.20),
                'topic_4': dict(ccc=0.35, bbb=0.53, aaa=0.12),
            }
        )
        assert (phi - real_topics).abs().values.max() < tolerance
    finally:
        shutil.rmtree(batches_folder)
