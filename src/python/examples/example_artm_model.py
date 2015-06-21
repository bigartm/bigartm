# Tests for new BigARTM Python API.
import artm.artm_model
from artm.artm_model import *

model = ArtmModel(num_processors=2, document_passes_count=10,
                  topic_names=['topic_1', 'topic_2', 'topic_3'], class_ids={'@default_class' : 1.0})
#                  topic_names=['top', 'top2', 'top3'], class_ids={'@default_class' : 1.0})
print model.topic_names

#model.parse(data_format='bow_uci', collection_name='kos')
model.initialize(dictionary=model.load_dictionary('kos/dictionary'))

model.scores.add(SparsityPhiScore(name='SpPhiScore'))
model.scores.add(SparsityThetaScore(name='SpThetaScore'))
model.scores.add(PerplexityScore(name='PerpScore'))
model.scores.add(ThetaSnippetScore(name='SnipScore', num_items=2))
model.scores.add(TopicKernelScore(name='KernelScore', probability_mass_threshold=0.9))
model.scores.add(ItemsProcessedScore(name='ItemScore'))

#model.regularizers.add(SmoothSparsePhiRegularizer('SmSpPhiReg', -0.1, ['@default_class'], ['topic_1', 'topic_2', 'topic_3']))
model.regularizers.add(SmoothSparseThetaRegularizer('SmSpThetaReg', -0.4))
model.regularizers.add(DecorrelatorPhiRegularizer('DecorPhiReg', 100000))

#model.load('artm_model')
#print model.topic_names

#model.fit_offline(data_path='kos', collection_passes_count=2)
#model.regularizers['SmSpPhiReg'].tau = -10
model.scores.add(TopTokensScore(name='TopTokenScore'))
model.scores['SnipScore'].num_items = 3
model.fit_offline(data_path='kos', collection_passes_count=2)

#model.fit_online(data_path='kos', batches=['066852ad-cf27-46a1-8f7b-c28b8db77ca9.batch',
#                                           'f2fdf7c1-92eb-4cad-9441-554f67190537.batch'])
#print model.scores_info['SpPhiScore'].value
#print '\n'
#print model.scores_info['SpThetaScore'].value
#print '\n'
#print model.scores_info['PerpScore'].value
#print '\n'
#for item in model.scores_info['SnipScore'].snippet:
#  print  item
#  print '\n'

#if not model.scores_info['TopTokenScore'].topic_info[0] is None:
#  print '\n'
#  print model.scores_info['TopTokenScore'].topic_info[0]['topic_1'].tokens
#  print model.scores_info['TopTokenScore'].topic_info[0]['topic_1'].weights
#  print model.scores_info['KernelScore'].topic_info[0]['topic_1'].tokens

#retval = model.find_theta(data_path='kos', batches=['a657484f-a028-4800-b965-f8ddf1d742f9.batch'])
#retval = model.get_theta(remove_theta=True)
#print retval.document_ids
#print len(retval[0])
#print retval[1]
#print model.scores_info['ItemScore'].value
#model.save()
#model.to_csv()
vis = model.visualize()
vis.to_file('lda.html')