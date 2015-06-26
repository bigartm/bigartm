# This is a simple example of basic usage of BigARTM's new Python API.

import glob

import artm.artm_model
from artm.artm_model import *

# Create the model objects
model = ArtmModel(num_processors=2, topics_count = 15, document_passes_count=1)

# Parse collection if it is necessary
if len(glob.glob('kos' + "/*.batch")) < 1:
    model.parse(data_path='', data_format='bow_uci', collection_name='kos')

# Initialize model using dictionary (e.g. fill create Phi matrix using information about
# vocabulary and topics count with random values in (0, 1))
model.initialize(dictionary=model.load_dictionary('kos/dictionary'))

# Create scores for model quality control
model.scores.add(SparsityPhiScore(name='SparsityPhiScore'))
model.scores.add(SparsityThetaScore(name='SparsityThetaScore'))
model.scores.add(PerplexityScore(name='PerplexityScore'))

# Create regularizers for model
model.regularizers.add(SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.3))
model.regularizers.add(SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.7))
model.regularizers.add(DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=100000.0))

# Learn model in offline mode (5 passes over whole collection)
model.fit_offline(data_path='kos', collection_passes_count=5)
num_phi_updates = 5

# Check sparsity
print 'Sparsity Phi:' + str(model.scores_info['SparsityPhiScore'].value[num_phi_updates-1])
print 'Sparsity Phi:' + str(model.scores_info['SparsityThetaScore'].value[num_phi_updates-1])

# Correct tau coefficients for regularizers to increase sparsity
model.regularizers['SparsePhi'].tau = -0.5
model.regularizers['SparseTheta'].tau = -0.9

# Add new score
model.scores.add(TopTokensScore(name='TopTokensScore', num_tokens=5))

# Continue learning model in offline mode (15 more passes)
model.fit_offline(data_path='kos', collection_passes_count=15)
num_phi_updates += 15

# Print all scores
print 'Per-iter score info:'
for i in range(num_phi_updates):
    print 'Iter# ' + str(i),
    print ', perplexity: %.2f' % model.scores_info['PerplexityScore'].value[i],
    print ', sparsity Phi: %.3f' % model.scores_info['SparsityPhiScore'].value[i],
    print ', sparsity Theta: %.3f,' % model.scores_info['SparsityThetaScore'].value[i]

print 'Final top tokens:'
for topic_name in model.topic_names:
    print topic_name + ': ',
    print model.scores_info['TopTokensScore'].topic_info[num_phi_updates-1][topic_name].tokens

# Save final model to model file and to .csv
model.save(file_name='kos_artm_model')
model.to_csv(file_name='kos_artm_model.csv')

vis = model.visualize(num_top_tokens=10, dictionary_path='kos/dictionary')
vis.to_file(filename='vis.html')