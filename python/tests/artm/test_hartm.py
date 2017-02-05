# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import tempfile
import os

import artm

def test_func():
    # constants
    num_tokens = 15
    parent_level_weight = 1
    num_collection_passes = 15
    num_document_passes = 10
    num_topics_level0 = 15
    num_topics_level1 = 50
    regularizer_tau = 10 ** 5
    vocab_size = 6906
    num_docs = 3430
    zero_eps = 0.001

    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')
    batches_folder = tempfile.mkdtemp()
    parent_batch_folder = tempfile.mkdtemp()

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=batch_vectorizer.data_path)

        hier = artm.hARTM(dictionary=dictionary, cache_theta=True, num_document_passes=num_document_passes)
        
        level0 = hier.add_level(num_topics=num_topics_level0)

        level0.initialize(dictionary=dictionary)
        
        level0.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)
        
        hier.tmp_files_path = parent_batch_folder
        level1 = hier.add_level(num_topics=num_topics_level1, parent_level_weight=parent_level_weight)
        
        level1.initialize(dictionary=dictionary)
        
        level1.regularizers.add(artm.HierarchySparsingThetaRegularizer(name="HierSp", tau=regularizer_tau))
        
        level1.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)

        phi = hier.get_level(1).get_phi()
        assert phi.shape == (vocab_size, num_topics_level1)
        # theta = hier.get_level(1).get_theta()
        # assert theta.shape == (num_topics_level1, num_docs)
        psi = hier.get_level(1).get_psi()
        support = psi.values.max(axis=1).min()

        # This test gives different results on python27 and python35. Authors need to investigate.
        on_python_27 = abs(support - 0.0978 < zero_eps)
        on_python_35 = abs(support - 0.1522 < zero_eps)
        assert(on_python_27 or on_python_35)
        
        assert(level1.clone() is not None)
        assert(hier.clone() is not None)
    finally:
        shutil.rmtree(batches_folder)
        shutil.rmtree(parent_batch_folder)
