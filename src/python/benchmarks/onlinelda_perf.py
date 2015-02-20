import os
import shutil
import json
import tempfile
import math
import subprocess

import gensim
import numpy as np


from performance import ResourceTracker, track_cmd_resource



def infer_stochastic_matrix(alphas, a=0, tol=1e-10):
    """
    Infer stochastic matrix from Dirichlet distribution.

    Input: matrix with rows corresponding to parameters of
    the asymmetric Dirichlet distributions, parameter a.

    a=0 => expected distributions
    a=1 => most probable distributions
    a=1/2 => normalized median-marginal distributions

    Returns: inferred stochastic matrix.
    """
    assert isinstance(alphas, np.ndarray)
    A = alphas - a
    A[A < tol] = 0
    A /= A.sum(axis=1, keepdims=True) + 1e-15
    return A


def compute_perplexity_artm(corpus, Phi, Theta):
    sum_n = 0.0
    sum_loglike = 0.0
    for doc_id, doc in enumerate(corpus):
        for term_id, count in doc:
            sum_n += count
            sum_loglike += count * np.log( np.dot(Theta[doc_id, :], Phi[:, term_id]) )
    perplexity = np.exp(- sum_loglike / sum_n)
    return perplexity


def start_report():
    try:
        uname = subprocess.check_output(['uname', '-a'])
    except OSError:
        uname = None
    try:
        lscpu = subprocess.check_output(['lscpu'])
    except OSError:
        lscpu = None

    report = {
        'machine': {
            'uname': uname,
            'lscpu': lscpu,
        },
    }
    return report


### Interface for OnlineLDA implementation

def run_impl(impl, name, train, test, wordids,
             num_processors=1, num_topics=100, batch_size=10000, passes=1,
             kappa=0.5, tau0=64, alpha=0.1, beta=0.1, num_inner_iters=20):
    """
    Run OnlineLDA algorithm.

    :param name: name of experiment
    :param train: train corpus
    :param test: dict {id: <test corpus>} where id is 'test' and 'valid'
    :param wordids: id-to-word mapping file (gensim Dictionary)
    :param num_processors: number of processors to use (more than one means parallelization)
    :param num_topics: number of topics
    :param batch_size: number of documents in the batch
    :param passes: number of passes through the train corpus
    :param kappa: power of learning rate
    :param tau0: initial learning rate
    :param alpha: document distribution smoothing coefficient (parameter dirichlet prior)
    :param beta: topic distribution smoothing coefficient (parameter dirichlet prior)
    :param num_inner_iters: maximal number of inner iterations on E-step
    """
    func = globals()['run_%s' % impl]
    func(name, train, test, wordids,
         num_processors=num_processors, num_topics=num_topics, batch_size=batch_size, passes=passes,
         kappa=kappa, tau0=tau0, alpha=alpha, beta=beta, num_inner_iters=num_inner_iters)


### Gensim
#
# Website: http://radimrehurek.com/gensim/
# Tutorial: http://radimrehurek.com/gensim/tut2.html
#

def run_gensim(name, train, test, wordids,
               num_processors=1, num_topics=100, batch_size=10000, passes=1,
               kappa=0.5, tau0=1.0, alpha=0.1, beta=0.1, update_every=1, num_inner_iters=20):

    if tau0 != 1.0:
        raise ValueError('Gensim does not support tau0 != 1.0')

    id2word = gensim.corpora.Dictionary.load_from_text('data/%s' % wordids)
    train_corpus_mm = gensim.corpora.MmCorpus('data/%s.mm' % train)

    # see https://github.com/piskvorky/gensim/issues/288
    from gensim.corpora.sharded_corpus import ShardedCorpus
    train_corpus = ShardedCorpus('data/%s.sc' % train, train_corpus_mm, shardsize=1000, overwrite=False)

    gamma_threshold = 0.001

    with ResourceTracker() as tracker:
        model = gensim.models.LdaMulticore(
            corpus=train_corpus,
            id2word=id2word,
            num_topics=num_topics,
            chunksize=batch_size,
            passes=passes,
            batch=False,
            alpha=alpha,
            eta=beta,
            decay=kappa,
            eval_every=0,
            iterations=num_inner_iters,
            gamma_threshold=gamma_threshold,
            workers=num_processors-1, # minus one because `workers`
                                      # is the number of extra processes
        )

    model.save('target/%s.gensim_model' % name)

    report = start_report()
    report['train_resources'] = tracker.report()

    Lambda = model.state.get_lambda()
    Phi = infer_stochastic_matrix(Lambda, 0)
    matrices = {
        'Lambda': Lambda,
        'Phi_mean': Phi,
        'Phi_map': infer_stochastic_matrix(Lambda, 1),
    }

    for id, corpus_name in test.iteritems():
        test_corpus = gensim.corpora.MmCorpus('data/%s.mm' % corpus_name)

        with ResourceTracker() as tracker:
            Gamma, _ = model.inference(test_corpus)

        Theta = infer_stochastic_matrix(Gamma, 0)
        matrices['%s_Gamma' % id] = Gamma
        matrices['%s_Theta_mean' % id] = Theta
        matrices['%s_Theta_map' % id] = infer_stochastic_matrix(Gamma, 1)

        report[id] = {
            'inference_resources': tracker.report(),
            'perplexity_gensim': np.exp(-model.log_perplexity(test_corpus)),
            'perplexity_artm': compute_perplexity_artm(test_corpus, Phi, Theta),
        }

    with open('target/%s.report.json' % name, 'w') as report_file:
        json.dump(report, report_file, indent=2)
    np.savez_compressed('target/%s.matrices.npz' % name, **matrices)


### Vowpal Wabbit
#
# Website: https://github.com/JohnLangford/vowpal_wabbit/wiki
# LDA Tutorial: https://github.com/JohnLangford/vowpal_wabbit/wiki/Latent-Dirichlet-Allocation
#

def run_vw(name, train, test, wordids,
           num_processors=1, num_topics=100, batch_size=10000, passes=1,
           kappa=0.5, tau0=64, alpha=0.1, beta=0.1, update_every=1, num_inner_iters=20, limit_docs=None, seed=123):

    def convert_mm_to_vw(mm_filename, vw_filename):
        with open(vw_filename, 'w') as fout, open(mm_filename) as fin:
            fin.readline()
            D, W, N = map(int, fin.readline().rstrip().split(' '))

            cur_d = None
            cur_feats = ''

            def emit_example():
                if cur_d is not None:
                    fout.write('0 \'%d | ' % cur_d + cur_feats + '\n')


            for line in fin:
                d, w, cnt = map(int, line.rstrip().split(' '))
                if cur_d != d:
                    emit_example()
                    cur_feats = ''
                    cur_d = d

                if cnt > 1:
                    cur_feats += ' %d:%d' % (w, cnt)
                else:
                    cur_feats += ' %d' % w
            emit_example()

    def read_vw_matrix(filename, topics=False, n_term=None):
        with open(filename) as f:
            if topics:
                for i in xrange(11): f.readline()
            result_matrix = []
            for line in f:
                parts = line.strip().replace('  ', ' ').split(' ')
                if topics:
                    index = int(parts[0])
                    matrix_line = map(float, parts[1:])
                    if index < n_term or not n_term:
                        result_matrix.append(matrix_line)
                else:
                    index = int(parts[-1])
                    matrix_line = map(float, parts[:-1])
                    result_matrix.append(matrix_line)
        return np.array(result_matrix, dtype=float)

    def read_vw_gammas(predictions_path):
        """
        Read matrix of inferred document distributions (gammas) from vw predictions file.
        :return: np.ndarray, size = num_docs x num_topics
        """
        gammas = read_vw_matrix(predictions_path, topics=False)
        return gammas

    def read_vw_lambdas(topics_path, n_term=None):
        """
        Read matrix of inferred topic distributions (lambdas) from vw readable model file.
        :param n_term: number of words
        :return: np.ndarray, size = num_topics x num_terms
        """
        lambdas = read_vw_matrix(topics_path, topics=True, n_term=n_term).T
        return lambdas

    if num_processors != 1:
        raise ValueError('Vowpal Wabbit LDA does not support parallelization')
    if update_every != 1:
        raise ValueError('Vowpal Wabbit LDA does not support update_every != 1')

    id2word = gensim.corpora.Dictionary.load_from_text('data/%s' % wordids)
    train_corpus = gensim.corpora.MmCorpus('data/%s.mm' % train)

    for name in [train] + test.values():
        if not os.path.exists('data/%s.vw' % name):
            print 'Converting %s.mm -> %s.vw' % (name, name)
            convert_mm_to_vw('data/%s.mm' % name, 'data/%s.vw' % name)

    tempdir = tempfile.mkdtemp()
    print 'Temp dir:', tempdir

    cmd = [
        'vw',
        'data/%s.vw' % train,
        '-b', '%.0f' % np.ceil(np.log2(len(id2word))),
        '--cache_file', os.path.join(tempdir, 'cache_file'),
        '--random_seed', str(seed),
        '--lda', str(num_topics),
        '--lda_alpha', str(alpha),
        '--lda_rho', str(beta),
        '--lda_D', str(train_corpus.num_docs),
        '--minibatch', str(batch_size),
        '--power_t', str(kappa),
        '--initial_t', str(tau0),
        '--passes', str(passes),
        '--readable_model', os.path.join(tempdir, 'readable_model'),
        '-p', os.path.join(tempdir, 'predictions'),
        '-f', 'target/%s.vw_model' % name,
    ]
    if limit_docs:
        cmd += ['--examples', str(limit_docs)]

    exitcode, tracker = track_cmd_resource(cmd)
    if exitcode != 0:
        raise RuntimeError('VW exited with non-zero code')

    report = start_report()
    report['train_resources'] = tracker.report()

    Lambda = read_vw_lambdas(os.path.join(tempdir, 'readable_model'), n_term=len(id2word))
    Phi = infer_stochastic_matrix(Lambda, 0)
    matrices = {
        'Lambda': Lambda,
        'Phi_mean': Phi,
        'Phi_map': infer_stochastic_matrix(Lambda, 1),
    }

    for id, corpus_name in test.iteritems():
        test_corpus = gensim.corpora.MmCorpus('data/%s.mm' % corpus_name)

        predictions_path = os.path.join(tempdir, 'predictions_%s' % id)
        cmd = [
            'vw',
            'data/%s.vw' % corpus_name,
            '--minibatch', str(test_corpus.num_docs),
            '--initial_regressor', 'target/%s.vw_model' % name,
            '-p', predictions_path,
        ]

        exitcode, tracker = track_cmd_resource(cmd)

        Gamma = read_vw_gammas(predictions_path)
        Theta = infer_stochastic_matrix(Gamma, 0)
        matrices['%s_Gamma' % id] = Gamma
        matrices['%s_Theta_mean' % id] = Theta
        matrices['%s_Theta_map' % id] = infer_stochastic_matrix(Gamma, 1)

        report[id] = {
            'inference_resources': tracker.report(),
            'perplexity_artm': compute_perplexity_artm(test_corpus, Phi, Theta),
        }

    with open('target/%s.report.json' % name, 'w') as report_file:
        json.dump(report, report_file, indent=2)
    np.savez_compressed('target/%s.matrices.npz' % name, **matrices)

    shutil.rmtree(tempdir)


### BigARTM
#
# Website: http://bigartm.org/
#


def run_bigartm(name, train, test, wordids,
                num_processors=1, num_topics=100, batch_size=10000, passes=1,
                kappa=0.5, tau0=64, alpha=0.1, beta=0.1, update_every=1, num_inner_iters=20):

    import artm.messages_pb2, artm.library

    def calc_theta_sparsity(theta_matrix):
        # theta_matrix should be an instance of messages_pb2.ThetaMatrix class
        # (http://docs.bigartm.org/en/latest/ref/messages.html#messages_pb2.ThetaMatrix)
        zeros = 0.0
        for item_index in range(0, len(theta_matrix.item_weights)):
            weights = theta_matrix.item_weights[item_index]
            for topic_index in range(0, len(weights.value)):
                if (weights.value[topic_index] < (0.001 / num_topics)):
                    zeros += 1.0
        return zeros / (len(theta_matrix.item_weights) * theta_matrix.topics_count)

    def prepare_batch_files(name, wordids, batch_size):
        batches_path = 'data/%s.bigartm_batches_%d' % (name, batch_size)
        if not os.path.exists(batches_path):
            subprocess.check_call([
                './bigartm_cpp_client',
                '--parsing_format', '1',
                '-v', 'data/%s' % wordids,
                '-d', 'data/%s.mm' % name,
                '--batch_folder', batches_path,
                '--items_per_batch', str(batch_size),
            ])
        return batches_path + '/'

    def calc_perplexity(topic_model, theta_matrix, batch):
        item_map = {}
        token_map = {}
        perplexity = 0.0
        perplexity_norm = 0.0
        for item_index in range(0, len(theta_matrix.item_id)):
            item_map[theta_matrix.item_id[item_index]] = item_index
        for token_index in range(0, len(topic_model.token)):
            token_map[topic_model.token[token_index]] = token_index
        for item in batch.item:
            if not item.id in item_map:
                raise Exception('Unable to find item_id=' + str(item.id) + ' in the theta matrix')
            theta_item_index = item_map[item.id]
            item_weights = theta_matrix.item_weights[theta_item_index].value
            field = item.field[0]
            for field_token_index in range(0, len(field.token_id)):
                batch_token_index = field.token_id[field_token_index]
                token_count = field.token_count[field_token_index]
                token = batch.token[batch_token_index]
                if not token in token_map:
                    raise Exception('Unable to find token=' + token + ' in the topic model')
                model_token_index = token_map[token]
                token_weights = topic_model.token_weights[model_token_index].value
                if len(token_weights) != len(item_weights):
                    raise Exception('Inconsistent topics count between Phi and Theta matrices')
                pwd = 0.0
                for topic_index in range(0, len(token_weights)):
                    pwd += token_weights[topic_index] * item_weights[topic_index]
                if pwd == 0:
                    raise Exception('Implement DocumentUnigramModel or CollectionUnigramModel to resolve p(w|d)=0 cases')
                perplexity += token_count * math.log(pwd)
                perplexity_norm += token_count
        return perplexity, perplexity_norm

    report = start_report()

    train_batches_folder = prepare_batch_files(train, wordids, batch_size)
    model_file_path = 'target/%s.bigartm_model' % name

    unique_tokens = artm.library.Library().LoadDictionary(train_batches_folder + 'dictionary')

    master_config = artm.messages_pb2.MasterComponentConfig()
    master_config.processors_count = num_processors
    master_config.cache_theta = True
    master_config.disk_path = train_batches_folder

    perplexity_collection_config = artm.messages_pb2.PerplexityScoreConfig()
    perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramCollectionModel
    perplexity_collection_config.dictionary_name = unique_tokens.name

    with artm.library.MasterComponent(master_config) as master:
        dictionary = master.CreateDictionary(unique_tokens)
        perplexity_score = master.CreatePerplexityScore(config = perplexity_collection_config)
        smooth_sparse_phi = master.CreateSmoothSparsePhiRegularizer()
        smooth_sparse_theta = master.CreateSmoothSparseThetaRegularizer()

        items_processed_score = master.CreateItemsProcessedScore()

        # Configure the model
        model = master.CreateModel(
            config=artm.messages_pb2.ModelConfig(),
            topics_count=num_topics,
            inner_iterations_count=num_inner_iters
        )
        model.EnableScore(items_processed_score)
        model.EnableRegularizer(smooth_sparse_phi, beta)
        model.EnableRegularizer(smooth_sparse_theta, alpha)

        model.Initialize(dictionary)

        with ResourceTracker() as tracker:

            master.InvokeIteration(passes)
            done = False
            first_sync = True
            next_items_processed = (batch_size * update_every)
            while (not done):
                done = master.WaitIdle(10)
                current_items_processed = items_processed_score.GetValue(model).value
                if done or (current_items_processed >= next_items_processed):
                    update_count = current_items_processed / (batch_size * update_every)
                    next_items_processed = current_items_processed + (batch_size * update_every)
                    rho = pow(tau0 + update_count, -kappa)
                    model.Synchronize(decay_weight=(0 if first_sync else (1-rho)), apply_weight=rho)
                    first_sync = False
                    print "Items processed: %i, Elapsed time: %.3f, Max memory: %.2f mb " % (
                        current_items_processed, tracker.elapsed_time, max(tracker.tick_mem))

        print 'Saving topic model... '
        with open(model_file_path, 'wb') as binary_file:
            binary_file.write(master.GetTopicModel(model).SerializeToString())

    report['train_resources'] = tracker.report()
    with open('target/%s.report.json' % name, 'w') as report_file:
        json.dump(report, report_file, indent=2)

    for test_key, test_name in test.iteritems():
        print 'Testing on hold-out set "%s"' % test_key

        report[test_key] = {}
        test_batches_folder = prepare_batch_files(test_name, wordids, batch_size)

        perplexity_collection_config = artm.messages_pb2.PerplexityScoreConfig()
        perplexity_collection_config.model_type = artm.library.PerplexityScoreConfig_Type_UnigramCollectionModel
        perplexity_collection_config.dictionary_name = unique_tokens.name

        test_master_config = artm.messages_pb2.MasterComponentConfig()
        test_master_config.processors_count = num_processors
        test_master_config.cache_theta = True
        test_master_config.disk_path = test_batches_folder

        with artm.library.MasterComponent(test_master_config) as test_master:
            print 'Loading topic model... '
            topic_model = artm.messages_pb2.TopicModel()
            with open(model_file_path, "rb") as binary_file:
                topic_model.ParseFromString(binary_file.read())

            test_dictionary = test_master.CreateDictionary(unique_tokens)
            test_perplexity_score = test_master.CreatePerplexityScore(config = perplexity_collection_config)
            smooth_sparse_phi = test_master.CreateSmoothSparsePhiRegularizer()
            smooth_sparse_theta = test_master.CreateSmoothSparseThetaRegularizer()

            test_model = test_master.CreateModel(topics_count = num_topics, inner_iterations_count = num_inner_iters)
            test_model.EnableScore(test_perplexity_score)
            test_model.EnableRegularizer(smooth_sparse_phi, beta)
            test_model.EnableRegularizer(smooth_sparse_theta, alpha)
            test_model.Overwrite(topic_model)

            # with TimeChecker() as timer:
            #     print 'Estimate perplexity on held out batches... '
            #     perplexity = 0.0; perplexity_norm = 1e-15
            #     for test_batch_filename in glob.glob(test_batches_folder + "*.batch"):
            #         print 'Test batch:', test_batch_filename
            #         test_batch = artm.library.Library().LoadBatch(test_batch_filename)
            #         test_batch_theta = test_master.GetThetaMatrix(model=test_model, batch=test_batch)
            #         theta_sparsity = calc_theta_sparsity(test_batch_theta)
            #         (batch_perplexity, batch_perplexity_norm) = calc_perplexity(topic_model, test_batch_theta, test_batch)
            #         print 'Batch = %s, Theta sparsity = %f, Perplexity = %f' % (
            #             test_batch_filename, theta_sparsity, math.exp(-batch_perplexity / batch_perplexity_norm))
            #         perplexity += batch_perplexity
            #         perplexity_norm += batch_perplexity_norm
            #     print 'Overall test perplexity = %f' % math.exp(-perplexity / perplexity_norm)
            #
            #     report[test_key]['inference_time'] = timer.status()
            #     report[test_key]['perplexity_artm'] = math.exp(-perplexity / perplexity_norm)

            with ResourceTracker() as tracker:
                test_master.InvokeIteration()
                test_master.WaitIdle()
                print "Test Perplexity calculated in BigARTM = %.3f" % test_perplexity_score.GetValue(test_model).value

                report[test_key]['inference_resources'] = tracker.report()
                report[test_key]['perplexity_bigartm'] = test_perplexity_score.GetValue(test_model).value

    with open('target/%s.report.json' % name, 'w') as report_file:
        json.dump(report, report_file, indent=2)
