import os
import sys 

#
# Path to BigARTM sources and build
#
BIGARTM_PATH = './bigartm'
BIGARTM_BUILD_PATH = './bigartm/build'

sys.path.append(os.path.join(BIGARTM_PATH, 'src/python'))
sys.path.append(os.path.join(BIGARTM_PATH, '3rdparty/protobuf/python'))
os.environ['ARTM_SHARED_LIBRARY'] = os.path.join(BIGARTM_BUILD_PATH, 'src/artm/libartm.so')
os.environ['LD_LIBRARY_PATH'] = os.path.join(BIGARTM_BUILD_PATH, 'src/artm/')

import onlinelda


if __name__ == '__main__':

    num_topics = 100
    alpha = 1.0 / num_topics
    beta = 1.0 / num_topics 

    for num_processors in [32, 24, 16, 12, 8, 4, 2, 1]:
        update_every = 160
        name = 'wiki_bigartm_parallel%d_batch10k_update%d' % (num_processors, update_every)
        if not os.path.exists('target/%s.report.json' % name):
             onlinelda.run_bigartm(
                name=name,
                train='wiki_bow_train',
                test={},
                wordids=None,
                num_processors=num_processors,
                num_topics=num_topics, alpha=alpha, beta=beta, batch_size=10000, update_every=update_every,
            )

