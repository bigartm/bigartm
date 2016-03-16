import shutil
import glob
import tempfile
import os
import pytest

import artm

def test_func():
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    batches_folder = tempfile.mkdtemp()
    try:
        data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        batch_vectorizer = None
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format='bow_uci',
                                                collection_name='kos',
                                                target_folder=batches_folder)

        model_artm = artm.ARTM(topic_names=['topic_{}'.format(i) for i in xrange(15)])

        model_artm.gather_dictionary('dictionary', batch_vectorizer.data_path)
        model_artm.initialize(dictionary_name='dictionary')

        model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.1))
        model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+5))

        model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
        model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore'))
        model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
        model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore'))
        model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore'))
        model_artm.scores.add(artm.ThetaSnippetScore(name='ThetaSnippetScore'))
    finally:
        shutil.rmtree(batches_folder)
