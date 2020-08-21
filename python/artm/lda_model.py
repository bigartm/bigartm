# Copyright 2017, Additive Regularization of Topic Models.

from copy import deepcopy

from six.moves import (
    range,
    zip,
)

from .artm_model import ARTM
from .regularizers import (
    SmoothSparsePhiRegularizer,
    SmoothSparseThetaRegularizer,
)
from .scores import (
    PerplexityScore,
    SparsityPhiScore,
    SparsityThetaScore,
    TopTokensScore,
)


class LDA(object):
    def __init__(self, num_topics=None, num_processors=None, cache_theta=False,
                 dictionary=None, num_document_passes=10, seed=-1, alpha=0.01,
                 beta=0.01, theta_columns_naming='id'):
        """
        :param int num_topics: the number of topics in model, will be overwrited if\
                                 topic_names is set
        :param int num_processors: how many threads will be used for model training, if\
                                 not specified then number of threads will be detected by the lib
        :param bool cache_theta: save or not the Theta matrix in model. Necessary if\
                                 ARTM.get_theta() usage expects
        :param int num_document_passes: number of inner iterations over each document
        :param dictionary: dictionary to be used for initialization, if None nothing will be done
        :type dictionary: str or reference to Dictionary object
        :param bool reuse_theta: reuse Theta from previous iteration or not
        :param seed: seed for random initialization, -1 means no seed
        :type seed: unsigned int or -1
        :param float alpha: hyperparameter of Theta smoothing regularizer
        :param beta: hyperparameter of Phi smoothing regularizer
        :type beta: float or list of floats with len == num_topics
        :param str theta_columns_naming: either 'id' or 'title', determines how to name columns\
                                 (documents) in theta dataframe

        :Note:
          * the type (not value!) of beta should not change after initialization: if it was scalar - it\
            should stay scalar, if it was list - it should stay list.
        """
        if not isinstance(alpha, float) and not isinstance(alpha, int):
            raise ValueError('LDA.alpha should be float')

        if not isinstance(beta, list) and not isinstance(beta, float) and not isinstance(beta, int):
            raise ValueError('LDA.beta should be float or list of floats')

        if isinstance(beta, list) and not (len(beta) == num_topics):
            raise ValueError('LDA.beta should have the length equal to num_topics')

        self._internal_model = ARTM(num_topics=num_topics,
                                    num_processors=num_processors,
                                    num_document_passes=num_document_passes,
                                    reuse_theta=True,
                                    cache_theta=cache_theta,
                                    dictionary=dictionary,
                                    seed=seed)

        self._dictionary = dictionary
        self._alpha = alpha
        self._beta = beta

        self._theta_reg_name = 'lda_theta_reg'
        self._phi_reg_name = 'lda_phi_reg'
        self._perp_score_name = 'perp_score'
        self._sp_phi_score_name = 'sp_phi_score'
        self._sp_theta_score_name = 'sp_theta_score'
        self._tt_score_name = 'tt_score'

        self._create_regularizers_and_scores()

    def clone(self):
        """
        :Description: returns a deep copy of the artm.LDA object

        :Note:
          * This method is equivalent to copy.deepcopy() of your artm.LDA object.
            For more information refer to artm.ARTM.clone() method.
        """
        return deepcopy(self)

    def _create_regularizers_and_scores(self):
        self._internal_model.regularizers.add(SmoothSparseThetaRegularizer(name=self._theta_reg_name, tau=self._alpha))
        if isinstance(self._beta, list):
            for i, b in enumerate(self._beta):
                phi_reg = SmoothSparsePhiRegularizer(name='{0}_{1}'.format(self._phi_reg_name, i), tau=b)
                self._internal_model.regularizers.add(phi_reg)
        else:
            self._internal_model.regularizers.add(SmoothSparsePhiRegularizer(name=self._phi_reg_name, tau=self._beta))

        if self._dictionary is None:
            self._internal_model.scores.add(PerplexityScore(name=self._perp_score_name))
        else:
            self._internal_model.scores.add(PerplexityScore(name=self._perp_score_name, dictionary=self._dictionary))

        self._internal_model.scores.add(SparsityThetaScore(name=self._sp_theta_score_name))
        self._internal_model.scores.add(SparsityPhiScore(name=self._sp_phi_score_name))

    def dispose(self):
        if self._internal_model is not None:
            self._internal_model.dispose()
            self._internal_model = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.dispose()

    def __del__(self):
        self.dispose()

    # ========== PROPERTIES ==========
    @property
    def num_topics(self):
        return self._internal_model.num_topics

    @property
    def num_processors(self):
        return self._internal_model.num_processors

    @property
    def cache_theta(self):
        return self._internal_model.cache_theta

    @property
    def dictionary(self):
        return self._internal_model.dictionary

    @property
    def num_document_passes(self):
        return self._internal_model.num_document_passes

    @property
    def seed(self):
        return self._internal_model.seed

    @property
    def theta_columns_naming(self):
        return self._internal_model.theta_columns_naming

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def phi_(self):
        return self._internal_model.phi_

    @property
    def perplexity_value(self):
        return self._internal_model.score_tracker[self._perp_score_name].value

    @property
    def perplexity_last_value(self):
        return self._internal_model.score_tracker[self._perp_score_name].last_value

    @property
    def sparsity_phi_value(self):
        return self._internal_model.score_tracker[self._sp_phi_score_name].value

    @property
    def sparsity_phi_last_value(self):
        return self._internal_model.score_tracker[self._sp_phi_score_name].last_value

    @property
    def sparsity_theta_value(self):
        return self._internal_model.score_tracker[self._sp_theta_score_name].value

    @property
    def sparsity_theta_last_value(self):
        return self._internal_model.score_tracker[self._sp_theta_score_name].last_value

    @property
    def library_version(self):
        return self._internal_model.library_version

    @property
    def master(self):
        return self._internal_model.master

    # ========== SETTERS ==========
    @num_processors.setter
    def num_processors(self, num_processors):
        self._internal_model.num_processors = num_processors

    @cache_theta.setter
    def cache_theta(self, cache_theta):
        self._internal_model.cache_theta = cache_theta

    @num_document_passes.setter
    def num_document_passes(self, num_document_passes):
        self._internal_model.num_document_passes = num_document_passes

    @seed.setter
    def seed(self, seed):
        self._internal_model.seed = seed

    @theta_columns_naming.setter
    def theta_columns_naming(self, theta_columns_naming):
        self._internal_model.theta_columns_naming = theta_columns_naming

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, float) and not isinstance(alpha, int):
            raise ValueError('LDA.alpha should be float')

        self._alpha = alpha
        self._internal_model.regularizers[self._theta_reg_name].tau = alpha

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, type(self._beta)):
            raise ValueError('LDA.beta shouldn\'t change type.')

        self._beta = beta
        if isinstance(beta, list):
            for i, b in enumerate(beta):
                self._internal_model.regularizers['{0}_{1}'.format(self._phi_reg_name, i)].tau = b
        else:
            self._internal_model.regularizers[self._phi_reg_name].tau = beta

    # ========== METHODS ==========
    def fit_offline(self, batch_vectorizer, num_collection_passes=1):
        """
        :Description: proceeds the learning of topic model in offline mode

        :param object_referenece batch_vectorizer: an instance of BatchVectorizer class
        :param int num_collection_passes: number of iterations over whole given collection
        """
        self._internal_model.fit_offline(batch_vectorizer=batch_vectorizer,
                                         num_collection_passes=num_collection_passes)

    def fit_online(self, batch_vectorizer, tau0=1024.0, kappa=0.7, update_every=1):
        """
        :Description: proceeds the learning of topic model in online mode

        :param object_reference batch_vectorizer: an instance of BatchVectorizer class
        :param int update_every: the number of batches; model will be updated once per it
        :param float tau0: coefficient (see 'Update formulas' paragraph)
        :param float kappa (float): power for tau0, (see 'Update formulas' paragraph)
        :param update_after: number of batches to be passed for Phi synchronizations
        :type update_after: list of int

        :Update formulas:
          * The formulas for decay_weight and apply_weight:
          * update_count = current_processed_docs / (batch_size * update_every);
          * rho = pow(tau0 + update_count, -kappa);
          * decay_weight = 1-rho;
          * apply_weight = rho;
        """
        self._internal_model.fit_online(batch_vectorizer=batch_vectorizer,
                                        tau0=tau0,
                                        kappa=kappa,
                                        update_every=update_every)

    def get_theta(self):
        """
        :Description: get Theta matrix for training set of documents

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the ids of documents, for which the Theta matrix was requested;
          * rows --- the names of topics in topic model, that was used to create Theta;
          * data --- content of Theta matrix.
        """
        if self._internal_model.cache_theta:
            return self._internal_model.get_theta()
        else:
            raise ValueError('cache_theta == False. Set LDA.cache_theta = True')

    def remove_theta(self):
        """
        :Description: removes cached theta matrix
        """
        self._internal_model.master.clear_theta_cache()

    def transform(self, batch_vectorizer, theta_matrix_type='dense_theta'):
        """
        :Description: find Theta matrix for new documents

        :param object_reference batch_vectorizer: an instance of BatchVectorizer class
        :param str theta_matrix_type: type of matrix to be returned, possible values:
                'dense_theta', None, default='dense_theta'

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the ids of documents, for which the Theta matrix was requested;
          * rows --- the names of topics in topic model, that was used to create Theta;
          * data --- content of Theta matrix.
        """
        return self._internal_model.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type=theta_matrix_type)

    def initialize(self, dictionary):
        """
        :Description: initialize topic model before learning

        :param dictionary: loaded BigARTM collection dictionary
        :type dictionary: str or reference to Dictionary object
        """
        if self._dictionary is None:
            self._dictionary = dictionary
            self._internal_model.scores[self._perp_score_name].use_unigram_document_model = False
            self._internal_model.scores[self._perp_score_name].dictionary = dictionary

        self._internal_model.initialize(dictionary=dictionary)

    def save(self, filename, model_name='p_wt'):
        """
        :Description: saves one Phi-like matrix to disk

        :param str filename: the name of file to store model
        :param str model_name: the name of matrix to be saved, 'p_wt' or 'n_wt'
        """
        self._internal_model.save(filename=filename, model_name=model_name)

    def load(self, filename, model_name='p_wt'):
        """
        :Description: loads from disk the topic model saved by LDA.save()

        :param str filename: the name of file containing model
        :param str model_name: the name of matrix to be saved, 'p_wt' or 'n_wt'

        :Note:
          * We strongly recommend you to reset all important parameters of the LDA\
            model, used earlier.
        """
        self._internal_model.load(filename=filename, model_name=model_name)
        if not len(self._internal_model.regularizers):
            self._create_regularizers_and_scores()

    def get_top_tokens(self, num_tokens=10, with_weights=False):
        """
        :Description: returns most probable tokens for each topic

        :param int num_tokens: number of top tokens to be returned
        :param bool with_weights: return only tokens, or tuples (token, its p_wt)

        :return:
          * list of lists of str, each internal list corresponds one topic in\
            natural order, if with_weights == False, or list, or list of lists\
            of tules, each tuple is (str, float)
        """
        self._internal_model.scores.add(
            TopTokensScore(name=self._tt_score_name, num_tokens=num_tokens), overwrite=True)
        result = self._internal_model.get_score(self._tt_score_name)

        tokens = []
        global_token_index = 0
        for topic_index in range(self.num_topics):
            if not with_weights:
                tokens.append(result.token[global_token_index: (global_token_index + num_tokens)])
            else:
                result_token = result.token[global_token_index: (global_token_index + num_tokens)]
                result_weight = result.weight[global_token_index: (global_token_index + num_tokens)]
                tokens.append(list(zip(result_token, result_weight)))
            global_token_index += num_tokens

        return tokens
