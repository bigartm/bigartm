# Copyright 2017, Additive Regularization of Topic Models.

import shutil
import tempfile
import os

import artm

def test_func():

    # constants
    num_documents = 3430
    vocabulary_size = 6906
    num_document_passes = 10
    num_collection_passes = 15
    num_topics_level_0 = 15
    num_topics_level_1 = 50
    parent_level_weight = 1
    regularizer_tau = 10 ** 5
    zero_eps = 0.001

    data_path = os.environ.get('BIGARTM_UNITTEST_DATA')

    batches_folder = tempfile.mkdtemp()
    parent_batch_folder = tempfile.mkdtemp()
    hierarchy_model_folder = tempfile.mkdtemp()

    try:
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=batch_vectorizer.data_path)

        hierarchy = artm.hARTM(dictionary=dictionary, cache_theta=True, num_document_passes=num_document_passes,
                               tmp_files_path=parent_batch_folder, theta_columns_naming="title")

        level_0 = hierarchy.add_level(num_topics=num_topics_level_0)
        level_0.initialize(dictionary=dictionary)
        level_0.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)

        phi_0 = hierarchy.get_level(0).get_phi()
        assert phi_0.shape == (vocabulary_size, num_topics_level_0)

        theta_0 = hierarchy.get_level(0).get_theta()
        assert theta_0.shape == (num_topics_level_0, num_documents)

        level_1 = hierarchy.add_level(num_topics=num_topics_level_1, parent_level_weight=parent_level_weight)
        level_1.initialize(dictionary=dictionary)
        level_1.regularizers.add(artm.HierarchySparsingThetaRegularizer(name="HierSparsTheta", tau=regularizer_tau))
        level_1.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=num_collection_passes)

        phi_1 = hierarchy.get_level(1).get_phi()
        assert phi_1.shape == (vocabulary_size, num_topics_level_1)

        theta_1 = hierarchy.get_level(1).get_theta()
        assert theta_1.shape == (num_topics_level_1, num_documents)

        psi = hierarchy.get_level(1).get_psi()
        assert psi.shape == (num_topics_level_1, num_topics_level_0)

        support = psi.values.max(axis=1).min()

        # This test gives different results on python27 and python35. Authors need to investigate.
        on_python_27 = abs(support - 0.0978 < zero_eps)
        on_python_35 = abs(support - 0.1522 < zero_eps)
        assert(on_python_27 or on_python_35)

        assert(level_0.clone() is not None)
        assert(level_1.clone() is not None)
        assert(hierarchy.clone() is not None)

        # Test the same functionality with hARTM, and validate that resulting psi matrix is exactly the same
        level_0_plain = artm.ARTM(num_topics=num_topics_level_0, num_document_passes=num_document_passes,
                                  cache_theta=True, seed=level_0.seed, theta_columns_naming="title")
        level_0_plain.initialize(dictionary=dictionary)
        level_0_plain.fit_offline(num_collection_passes=num_collection_passes, batch_vectorizer=batch_vectorizer)

        phi_0_plain = level_0_plain.get_phi()
        assert (phi_0 - phi_0_plain).abs().max().max() < 1e-3

        theta_0_plain = level_0_plain.get_theta()
        assert (theta_0 - theta_0_plain).abs().max().max() < 1e-3

        level_1_plain = artm.ARTM(num_topics=num_topics_level_1, num_document_passes=num_document_passes,
                                  parent_model=level_0_plain, parent_model_weight=parent_level_weight,
                                  cache_theta=True, seed=level_1.seed, theta_columns_naming="title")
        level_1_plain.initialize(dictionary=dictionary)
        level_1_plain.regularizers.add(artm.HierarchySparsingThetaRegularizer(name="HierSparsTheta", tau=regularizer_tau))
        level_1_plain.fit_offline(num_collection_passes=num_collection_passes, batch_vectorizer=batch_vectorizer)

        phi_1_plain = level_1_plain.get_phi()
        assert (phi_1 - phi_1_plain).abs().max().max() < 1e-3

        theta_1_plain = level_1_plain.get_theta()
        assert (theta_1 - theta_1_plain).abs().max().max() < 1e-3

        psi_plain = level_1_plain.get_parent_psi()
        assert (psi - psi_plain).abs().max().max() < 1e-3

        #Test save and load methods

        hierarchy.save(hierarchy_model_folder)

        hierarchy_load = artm.hARTM()
        hierarchy_load.load(hierarchy_model_folder)

        assert level_0.num_topics == hierarchy_load.get_level(0).num_topics
        assert (phi_0 - hierarchy_load.get_level(0).get_phi()).abs().max().max() < 1e-3

        assert level_1.num_topics == hierarchy_load.get_level(1).num_topics
        assert (phi_1 - hierarchy_load.get_level(1).get_phi()).abs().max().max() < 1e-3

    finally:
        shutil.rmtree(batches_folder)
        shutil.rmtree(parent_batch_folder)
        shutil.rmtree(hierarchy_model_folder)