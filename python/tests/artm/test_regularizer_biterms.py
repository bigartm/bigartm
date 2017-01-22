# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import glob
import tempfile
import os
import pytest
import uuid

from six.moves import range

import artm


def test_func():
    biterms_tau = 0.0
    num_collection_passes = 1
    num_document_passes = 1
    num_topics = 3
    phi_first_elem = 0.2109  # check that initialization had not changed
    phi_eps = 0.0001

    batches_folder = tempfile.mkdtemp()
    vocab_file_name = os.path.join(batches_folder, 'vocab.txt')
    cooc_file_name = cooc_file_path=os.path.join(batches_folder, 'cooc_data.txt')

    phi_values = [[0.380308, 0.659777, 0.429884],
                  [0.330372, 0.012429, 0.081726],
                  [0.277840, 0.020186, 0.334808],
                  [0.011480, 0.307608, 0.153582]]

    try:
        batch = artm.messages.Batch()
        batch.token.append('A')
        batch.token.append('B')
        batch.token.append('C')
        batch.token.append('D')

        item = batch.item.add()
        item.token_id.append(0)
        item.token_id.append(2)
        item.token_id.append(3)
        item.token_id.append(0)

        item.token_weight.append(2)
        item.token_weight.append(4)
        item.token_weight.append(1)
        item.token_weight.append(1)

        item = batch.item.add()
        item.token_id.append(1)
        item.token_id.append(2)
        item.token_id.append(0)
        item.token_id.append(3)

        item.token_weight.append(3)
        item.token_weight.append(2)
        item.token_weight.append(4)
        item.token_weight.append(1)

        with open(os.path.join(batches_folder, '{}.batch'.format(uuid.uuid4())), 'wb') as fout:
            fout.write(batch.SerializeToString())

        batch = artm.messages.Batch()
        batch.token.append('A')
        batch.token.append('B')
        batch.token.append('D')

        item = batch.item.add()
        item.token_id.append(0)
        item.token_id.append(1)
        item.token_id.append(2)

        item.token_weight.append(2)
        item.token_weight.append(1)
        item.token_weight.append(1)

        item = batch.item.add()
        item.token_id.append(0)
        item.token_id.append(2)

        item.token_weight.append(6)
        item.token_weight.append(2)

        with open(os.path.join(batches_folder, '{}.batch'.format(uuid.uuid4())), 'wb') as fout:
            fout.write(batch.SerializeToString())

        with open(vocab_file_name, 'w') as fout:
            for e in ['A', 'B', 'C', 'D']:
                fout.write('{0}\n'.format(e))

        with open(cooc_file_name, 'w') as fout:
            fout.write('0 3 5.0\n')
            fout.write('0 1 4.0\n')
            fout.write('0 2 5.0\n')
            fout.write('1 3 2.0\n')
            fout.write('1 2 2.0\n')
            fout.write('2 3 2.0\n')

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=batches_folder, vocab_file_path=vocab_file_name, cooc_file_path=cooc_file_name)
        batch_vectorizer = artm.BatchVectorizer(data_path=batches_folder, data_format='batches')

        model = artm.ARTM(num_topics=num_topics, dictionary=dictionary, num_document_passes=num_document_passes)
        model.regularizers.add(artm.BitermsPhiRegularizer(name='Biterms', tau=biterms_tau, dictionary=dictionary))

        assert abs(model.phi_.as_matrix()[0][0] - phi_first_elem) < phi_eps
    
        model.fit_offline(batch_vectorizer=batch_vectorizer)
        for i in range(len(phi_values)):
            for j in range(len(phi_values[0])):
                assert abs(model.phi_.as_matrix()[i][j] - phi_values[i][j]) < phi_eps
    finally:
        shutil.rmtree(batches_folder)
