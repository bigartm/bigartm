# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import pytest

from six.moves import range

import artm

def test_func():
    def generate_ground_truth():
        doc_to_transactions = {}
        for i in range(num_docs):
            doc_to_transactions[i] = []

        doc_to_transactions[0].append([('class_1', 'token_1', '@default_transaction')])
        doc_to_transactions[0].append([('class_1', 'token_1', 'trans1'), ('class_2', 'token_2', 'trans1' )])

        doc_to_transactions[1].append([('class_1', 'token_2', '@default_transaction')])
        doc_to_transactions[1].append([('class_1', 'token_2', 'trans1' ), ('class_2', 'token_3', 'trans1')])

        doc_to_transactions[2].append([('class_1', 'token_3', '@default_transaction')])
        doc_to_transactions[2].append([('class_1', 'token_3', 'trans1'), ('class_2', 'token_4', 'trans1' )])

        doc_to_transactions[3].append([('class_1', 'token_1', '@default_transaction')])
        doc_to_transactions[3].append([('class_1', 'token_1', 'trans1'), ('class_2', 'token_2', 'trans1' )])

        doc_to_transactions[4].append([('class_1', 'token_2', '@default_transaction')])
        doc_to_transactions[4].append([('class_1', 'token_2', 'trans1'), ('class_2', 'token_3', 'trans1')])

        doc_to_transactions[5].append([('class_1', 'token_3', '@default_transaction')])
        doc_to_transactions[5].append([('class_1', 'token_3', 'trans1'), ('class_2', 'token_4', 'trans1')])
  
        doc_to_transactions[6].append([('class_3', 'token_5', '@default_transaction')])
        doc_to_transactions[6].append([('class_4', 'token_5', 'trans2'), ('class_2', 'token_2', 'trans2'), ('class_1', 'token_2', 'trans2')])

        doc_to_transactions[7].append([('class_1', 'token_1', 'trans1'), ('class_2', 'token_2', 'trans1')])
        doc_to_transactions[7].append([('class_1', 'token_2', 'trans1'), ('class_2', 'token_3', 'trans1')])
        doc_to_transactions[7].append([('class_1', 'token_1', '@default_transaction')])

        return doc_to_transactions


    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()

    num_topics = 3
    num_docs = 8
    num_tokens = 17

    try:
        bv = artm.BatchVectorizer(data_path=os.path.join(data_path,
                                  'vw_transaction_data_extended.txt'),
                                  data_format='vowpal_wabbit',
                                  target_folder=batches_folder)

        model = artm.ARTM(num_topics=num_topics,
                          cache_theta=True,
                          class_ids={'class_1': 1.0, 'class_2': 1.0, 'class_3': 1.0, 'class_4': 1.0},
                          transaction_typenames={'@default_transaction': 1.0, 'trans1': 1.0, 'trans2': 1.0},
                          dictionary=bv.dictionary)
        model.scores.add(artm.PerplexityScore(name='PerplexityScore', dictionary=bv.dictionary))

        doc_to_transactions = generate_ground_truth()

        num_iters = 5
        model.fit_offline(bv, num_collection_passes=num_iters)
        values = model.score_tracker['PerplexityScore'].value + [-1.0]
        for i in range(num_iters - 1):
            assert values[i] > values[i + 1]

        phi = model.get_phi()
        theta = model.get_theta()

        assert len(phi.columns) == num_topics
        assert len(phi.index) == num_tokens
        assert len(theta.index) == num_topics
        assert len(theta.columns) == num_docs

        for i_d, d in enumerate(theta.columns):
            transactions = doc_to_transactions[d]
            for i_x, x in enumerate(transactions):
                p_xd = 0.0
                for t in phi.columns:
                    val = theta[d][t]
                    for tok in x:
                        val *= phi[t][tok]
                    p_xd += val

                if i_d != 7:
                    assert abs(p_xd - 1.0) < 0.01
                elif i_x == 0 or i_x == 2:
                    assert abs(p_xd - 0.66) < 0.01
                elif i_x == 1:
                    assert abs(p_xd - 0.33) < 0.01
                else:
                   raise RuntimeError("Invalid i_x or i_d: {}, {}".format(i_x, i_d))
    finally:
        shutil.rmtree(batches_folder)

