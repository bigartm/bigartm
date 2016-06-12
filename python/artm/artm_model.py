import os
import csv
import uuid
import glob
import shutil
import tempfile

from pandas import DataFrame

from . import wrapper
from wrapper import constants as const
from wrapper import messages_pb2
from . import master_component as mc

from .regularizers import Regularizers
from .scores import Scores, TopicMassPhiScore  # temp
from . import score_tracker

SCORE_TRACKER = {
    const.ScoreType_SparsityPhi: score_tracker.SparsityPhiScoreTracker,
    const.ScoreType_SparsityTheta: score_tracker.SparsityThetaScoreTracker,
    const.ScoreType_Perplexity: score_tracker.PerplexityScoreTracker,
    const.ScoreType_ThetaSnippet: score_tracker.ThetaSnippetScoreTracker,
    const.ScoreType_ItemsProcessed: score_tracker.ItemsProcessedScoreTracker,
    const.ScoreType_TopTokens: score_tracker.TopTokensScoreTracker,
    const.ScoreType_TopicKernel: score_tracker.TopicKernelScoreTracker,
    const.ScoreType_TopicMassPhi: score_tracker.TopicMassPhiScoreTracker,
    const.ScoreType_ClassPrecision: score_tracker.ClassPrecisionScoreTracker,
    const.ScoreType_BackgroundTokensRatio: score_tracker.BackgroundTokensRatioScoreTracker,
}


def _topic_selection_regularizer_func(self, regularizers):
    topic_selection_regularizer_name = []
    for name, regularizer in regularizers.data.iteritems():
        if regularizer.type == const.RegularizerType_TopicSelectionTheta:
            topic_selection_regularizer_name.append(name)

    if len(topic_selection_regularizer_name):
        n_t = [0] * self.num_topics
        no_score = self._internal_topic_mass_score_name is None
        if no_score:
            self._internal_topic_mass_score_name = 'ITMScore_{}'.format(str(uuid.uuid4()))
            self.scores.add(TopicMassPhiScore(name=self._internal_topic_mass_score_name,
                                              class_id='@default_class'))  # ugly hack!

        if not self._synchronizations_processed or no_score:
            phi = self.get_phi(class_ids=['@default_class'])  # ugly hack!
            n_t = list(phi.sum(axis=0))
        else:
            for i, n in enumerate(self.topic_names):
                n_t[i] = self.score_tracker[
                    self._internal_topic_mass_score_name].last_topic_mass[n]

        n = sum(n_t)
        for name in topic_selection_regularizer_name:
            config = self.regularizers[name]._config_message()
            config.CopyFrom(self.regularizers[name].config)
            config.ClearField('topic_value')
            for value in [n / (e * self.num_topics) if e > 0.0 else 0.0 for e in n_t]:
                config.topic_value.append(value)
            self.regularizers[name].config = config


class ARTM(object):
    def __init__(self, num_topics=None, topic_names=None, num_processors=None, class_ids=None,
                 scores=None, regularizers=None, num_document_passes=10, reuse_theta=False,
                 dictionary=None, cache_theta=False, theta_columns_naming='id', seed=-1):
        """
        :param int num_topics: the number of topics in model, will be overwrited if\
                                 topic_names is set
        :param int num_processors: how many threads will be used for model training, if\
                                 not specified then number of threads will be detected by the lib
        :param topic_names: names of topics in model
        :type topic_names: list of str
        :param dict class_ids: list of class_ids and their weights to be used in model,\
                                 key --- class_id, value --- weight, if not specified then\
                                 all class_ids will be used
        :param bool cache_theta: save or not the Theta matrix in model. Necessary if\
                                 ARTM.get_theta() usage expects
        :param list scores: list of scores (objects of artm.*Score classes)
        :param list regularizers: list with regularizers (objects of artm.*Regularizer classes)
        :param int num_document_passes: number of inner iterations over each document
        :param dictionary: dictionary to be used for initialization, if None nothing will be done
        :type dictionary: str or reference to Dictionary object
        :param bool reuse_theta: reuse Theta from previous iteration or not
        :param str theta_columns_naming: either 'id' or 'title', determines how to name columns\
                                 (documents) in theta dataframe
        :param seed: seed for random initialization, -1 means no seed
        :type seed: unsigned int or -1

        :Important public fields:
          * regularizers: contains dict of regularizers, included into model
          * scores: contains dict of scores, included into model
          * score_tracker: contains dict of scoring results:\
               key --- score name, value --- ScoreTracker object, which contains info about\
               values of score on each synchronization (e.g. collection pass) in list

        :Note:
          * Here and anywhere in BigARTM empty topic_names or class_ids means that\
            model (or regularizer, or score) should use all topics or class_ids.
          * If some fields of regularizers or scores are not defined by\
            user --- internal lib defaults would be used.
          * If field 'topic_names' is None, it will be generated by BigARTM and will\
            be available using ARTM.topic_names().
        """
        self._num_processors = None
        self._cache_theta = False
        self._num_document_passes = num_document_passes
        self._reuse_theta = True
        self._theta_columns_naming = 'id'
        self._seed = -1

        if topic_names is not None:
            self._topic_names = topic_names
        elif num_topics is not None:
            self._topic_names = ['topic_{}'.format(i) for i in xrange(num_topics)]
        else:
            raise ValueError('Either num_topics or topic_names parameter should be set')

        if class_ids is None:
            self._class_ids = {}
        elif len(class_ids) > 0:
            self._class_ids = class_ids

        if isinstance(num_processors, int) and num_processors > 0:
            self._num_processors = num_processors

        if isinstance(cache_theta, bool):
            self._cache_theta = cache_theta

        if isinstance(reuse_theta, bool):
            self._reuse_theta = reuse_theta

        if isinstance(num_document_passes, int):
            self._num_document_passes = num_document_passes

        if theta_columns_naming in ['id', 'title']:
            self._theta_columns_naming = theta_columns_naming

        if isinstance(seed, int) and seed >= 0:
            self._seed = seed

        self._model_pwt = 'pwt'
        self._model_nwt = 'nwt'

        self._lib = wrapper.LibArtm()
        self._master = mc.MasterComponent(self._lib,
                                          num_processors=self._num_processors,
                                          topic_names=self._topic_names,
                                          class_ids=self._class_ids,
                                          pwt_name=self._model_pwt,
                                          nwt_name=self._model_nwt,
                                          num_document_passes=self._num_document_passes,
                                          reuse_theta=self._reuse_theta,
                                          cache_theta=self._cache_theta)

        self._regularizers = Regularizers(self._master)
        self._scores = Scores(self._master, self._model_pwt, self._model_nwt)

        # add scores and regularizers if necessary
        if scores is not None:
            for score in scores:
                self._scores.add(score)
        if regularizers is not None:
            for regularizer in regularizers:
                self._regularizers.add(regularizer)

        self._score_tracker = {}
        self._synchronizations_processed = 0
        self._initialized = False
        self._phi_cached = None  # This field will be set during .phi_ call
        self._num_online_processed_batches = 0

        # temp code for easy using of TopicSelectionThetaRegularizer from Python
        self._internal_topic_mass_score_name = None

        if dictionary is not None:
            self.initialize(dictionary)

    def __enter__(self):
        return self

    def dispose(self):
        """
        :Description: free all native memory, allocated for this model

        :Note:
          * This method does not free memory occupied by dictionaries,
            because dictionaries are shared across all models
          * ARTM class implements __exit__ and __del___ methods,
            which automatically call dispose.
        """
        if self._master is not None:
            self._lib.ArtmDisposeMasterComponent(self.master.master_id)
            self._master = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.dispose()

    def __del__(self):
        self.dispose()

    # ========== PROPERTIES ==========
    @property
    def num_processors(self):
        return self._num_processors

    @property
    def cache_theta(self):
        return self._cache_theta

    @property
    def reuse_theta(self):
        return self._reuse_theta

    @property
    def num_document_passes(self):
        return self._num_document_passes

    @property
    def theta_columns_naming(self):
        return self._theta_columns_naming

    @property
    def num_topics(self):
        return len(self._topic_names)

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def regularizers(self):
        return self._regularizers

    @property
    def scores(self):
        return self._scores

    @property
    def score_tracker(self):
        return self._score_tracker

    @property
    def master(self):
        return self._master

    @property
    def model_pwt(self):
        return self._model_pwt

    @property
    def model_nwt(self):
        return self._model_nwt

    @property
    def num_phi_updates(self):
        return self._synchronizations_processed

    @property
    def num_online_processed_batches(self):
        return self._num_online_processed_batches

    @property
    def seed(self):
        return self._seed

    @property
    def phi_(self):
        if self._phi_cached is None:
            self._phi_cached = self.get_phi()
        return self._phi_cached

    @property
    def info(self):
        """
        :Description: returns internal diagnostics information about the model
        """
        return self.master.get_info()

    @property
    def library_version(self):
        """
        :Description: the version of BigARTM library in a MAJOR.MINOR.PATCH format
        """
        return self._lib.version()

    # ========== SETTERS ==========
    @num_processors.setter
    def num_processors(self, num_processors):
        if num_processors <= 0 or not isinstance(num_processors, int):
            raise IOError('Number of processors should be a positive integer')
        else:
            self.master.reconfigure(num_processors=num_processors)
            self._num_processors = num_processors

    @cache_theta.setter
    def cache_theta(self, cache_theta):
        if not isinstance(cache_theta, bool):
            raise IOError('cache_theta should be bool')
        else:
            self.master.reconfigure(cache_theta=cache_theta)
            self._cache_theta = cache_theta

    @reuse_theta.setter
    def reuse_theta(self, reuse_theta):
        if not isinstance(reuse_theta, bool):
            raise IOError('reuse_theta should be bool')
        else:
            self.master.reconfigure(reuse_theta=reuse_theta)
            self._reuse_theta = reuse_theta

    @num_online_processed_batches.setter
    def num_online_processed_batches(self, num_online_processed_batches):
        if num_online_processed_batches <= 0 or not isinstance(num_online_processed_batches, int):
            raise IOError('Number of processed batches should be a positive integer')
        else:
            self._num_online_processed_batches = num_online_processed_batches

    @num_document_passes.setter
    def num_document_passes(self, num_document_passes):
        if num_document_passes <= 0 or not isinstance(num_document_passes, int):
            raise IOError('Number of passes through document should be a positive integer')
        else:
            self.master.reconfigure(num_document_passes=num_document_passes)
            self._num_document_passes = num_document_passes

    @theta_columns_naming.setter
    def theta_columns_naming(self, theta_columns_naming):
        if theta_columns_naming not in ['id', 'title']:
            raise IOError('theta_columns_naming should be either id or title')
        else:
            self._theta_columns_naming = theta_columns_naming

    @topic_names.setter
    def topic_names(self, topic_names):
        if not topic_names:
            raise IOError('Number of topic names should be non-negative')
        else:
            self.master.reconfigure(topic_names=topic_names)
            self._topic_names = topic_names

    @class_ids.setter
    def class_ids(self, class_ids):
        if len(class_ids) < 0:
            raise IOError('Number of (class_id, class_weight) pairs should be non-negative')
        else:
            self.master.reconfigure(class_ids=class_ids)
            self._class_ids = class_ids

    @seed.setter
    def seed(self, seed):
        if seed < 0 or not isinstance(seed, int):
            raise IOError('Random seed should be a positive integer')
        else:
            self._seed = seed

    # ========== METHODS ==========
    def fit_offline(self, batch_vectorizer=None, num_collection_passes=1):
        """
        :Description: proceeds the learning of topic model in offline mode

        :param object_referenece batch_vectorizer: an instance of BatchVectorizer class
        :param int num_collection_passes: number of iterations over whole given collection
        """
        if batch_vectorizer is None:
            raise IOError('No batches were given for processing')

        if not self._initialized:
            raise RuntimeError('The model was not initialized. Use initialize() method')

        batches_list = [batch.filename for batch in batch_vectorizer.batches_list]
        # outer cycle is needed because of TopicSelectionThetaRegularizer
        # and current ScoreTracker implementation
        for _ in xrange(num_collection_passes):
            # temp code for easy using of TopicSelectionThetaRegularizer from Python
            _topic_selection_regularizer_func(self, self._regularizers)

            self._synchronizations_processed += 1
            self.master.fit_offline(batch_filenames=batches_list,
                                    batch_weights=batch_vectorizer.weights,
                                    num_collection_passes=1)

            for name in self.scores.data.keys():
                if name not in self.score_tracker:
                    self.score_tracker[name] =\
                        SCORE_TRACKER[self.scores[name].type](self.scores[name])

        self._phi_cached = None

    def fit_online(self, batch_vectorizer=None, tau0=1024.0, kappa=0.7, update_every=1,
                   apply_weight=None, decay_weight=None, update_after=None, async=False):
        """
        :Description: proceeds the learning of topic model in online mode

        :param object_reference batch_vectorizer: an instance of BatchVectorizer class
        :param int update_every: the number of batches; model will be updated once per it
        :param float tau0: coefficient (see 'Update formulas' paragraph)
        :param float kappa (float): power for tau0, (see 'Update formulas' paragraph)
        :param update_after: number of batches to be passed for Phi synchronizations
        :type update_after: list of int
        :param apply_weight: weight of applying new counters
        :type apply_weight: list of float
        :param decay_weight: weight of applying old counters
        :type decay_weight: list of float
        :param bool async: use or not the async implementation of the EM-algorithm

        :Note:
          async=True leads to impossibility of score extraction via score_tracker.\
          Use get_score() instead.

        :Update formulas:
          * The formulas for decay_weight and apply_weight:
          * update_count = current_processed_docs / (batch_size * update_every);
          * rho = pow(tau0 + update_count, -kappa);
          * decay_weight = 1-rho;
          * apply_weight = rho;
          * if apply_weight, decay_weight and update_after are set, they will be used,\
            otherwise the code below will be used (with update_every, tau0 and kappa)
        """
        if batch_vectorizer is None:
            raise IOError('No batches were given for processing')

        if not self._initialized:
            raise RuntimeError('The model was not initialized. Use initialize() method')

        batches_list = [batch.filename for batch in batch_vectorizer.batches_list]

        update_after_final, apply_weight_final, decay_weight_final = [], [], []
        if (update_after is None) or (apply_weight is None) or (decay_weight is None):
            update_after_final = range(update_every, batch_vectorizer.num_batches + 1, update_every)
            if len(update_after_final) == 0 or (update_after_final[-1] != batch_vectorizer.num_batches):
                update_after_final.append(batch_vectorizer.num_batches)

            for _ in update_after_final:
                self._num_online_processed_batches += update_every
                update_count = self._num_online_processed_batches / update_every
                rho = pow(tau0 + update_count, -kappa)
                apply_weight_final.append(rho)
                decay_weight_final.append(1 - rho)
        else:
            update_after_final = update_after
            apply_weight_final = apply_weight
            decay_weight_final = decay_weight

        # temp code for easy using of TopicSelectionThetaRegularizer from Python
        _topic_selection_regularizer_func(self, self._regularizers)

        self.master.fit_online(batch_filenames=batches_list,
                               batch_weights=batch_vectorizer.weights,
                               update_after=update_after_final,
                               apply_weight=apply_weight_final,
                               decay_weight=decay_weight_final,
                               async=async)

        for name in self.scores.data.keys():
            if name not in self.score_tracker:
                self.score_tracker[name] =\
                    SCORE_TRACKER[self.scores[name].type](self.scores[name])

        self._synchronizations_processed += len(update_after_final)
        self._phi_cached = None

    def save(self, filename, model_name='p_wt'):
        """
        :Description: saves one Phi-like matrix to disk

        :param str filename: the name of file to store model
        :param str model_name: the name of matrix to be saved, 'p_wt' or 'n_wt'
        """
        if not self._initialized:
            raise RuntimeError('Model does not exist yet. Use ARTM.initialize()/ARTM.fit_*()')

        if os.path.isfile(filename):
            os.remove(filename)

        _model_name = None
        if model_name == 'p_wt':
            _model_name = self.model_pwt
        elif model_name == 'n_wt':
            _model_name = self.model_nwt

        self.master.export_model(_model_name, filename)

    def load(self, filename, model_name='p_wt'):
        """
        :Description: loads from disk the topic model saved by ARTM.save()

        :param str filename: the name of file containing model
        :param str model_name: the name of matrix to be saved, 'p_wt' or 'n_wt'

        :Note:
          * Loaded model will overwrite ARTM.topic_names and class_ids fields.
          * All class_ids weights will be set to 1.0, you need to specify them by\
            hand if it's necessary.
          * The method call will empty ARTM.score_tracker.
          * All regularizers and scores will be forgotten.
          * etc.
          * We strongly recommend you to reset all important parameters of the ARTM\
            model, used earlier.
        """
        _model_name = None
        if model_name == 'p_wt':
            _model_name = self.model_pwt
        elif model_name == 'n_wt':
            _model_name = self.model_nwt

        self.master.import_model(_model_name, filename)
        self._initialized = True
        topics_and_tokens_info = self.master.get_phi_info(self.model_pwt)
        self._topic_names = [topic_name for topic_name in topics_and_tokens_info.topic_name]

        class_ids = {}
        for class_id in topics_and_tokens_info.class_id:
            class_ids[class_id] = 1.0
        self._class_ids = class_ids

        # Remove all info about previous iterations
        self._score_tracker = {}
        self._synchronizations_processed = 0
        self._num_online_processed_batches = 0
        self._phi_cached = None

    def get_phi(self, topic_names=None, class_ids=None, model_name=None):
        """
        :Description: get custom Phi matrix of model. The extraction of the\
                      whole Phi matrix expects ARTM.phi_ call.

        :param topic_names: list with topics to extract, None value means all topics
        :type topic_names: list of str
        :param class_ids: list with class ids to extract, None means all class ids
        :type class_ids: list of str
        :param str model_name: self.model_pwt by default, self.model_nwt is also\
                      reasonable to extract unnormalized counters

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the names of topics in topic model;
          * rows --- the tokens of topic model;
          * data --- content of Phi matrix.
        """
        if not self._initialized:
            raise RuntimeError('Model does not exist yet. Use ARTM.initialize()/ARTM.fit_*()')

        valid_model_name = self.model_pwt if model_name is None else model_name

        topics_and_tokens_info = self.master.get_phi_info(valid_model_name)

        _, nd_array = self.master.get_phi_matrix(model=valid_model_name,
                                                 topic_names=topic_names,
                                                 class_ids=class_ids)

        tokens = [token for token, class_id in zip(topics_and_tokens_info.token, topics_and_tokens_info.class_id)
                  if class_ids is None or class_id in class_ids]
        topic_names = [topic_name for topic_name in topics_and_tokens_info.topic_name
                       if topic_names is None or topic_name in topic_names]
        phi_data_frame = DataFrame(data=nd_array,
                                   columns=topic_names,
                                   index=tokens)

        return phi_data_frame

    def get_theta(self, topic_names=None):
        """
        :Description: get Theta matrix for training set of documents

        :param topic_names: list with topics to extract, None means all topics
        :type topic_names: list of str

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the ids of documents, for which the Theta matrix was requested;
          * rows --- the names of topics in topic model, that was used to create Theta;
          * data --- content of Theta matrix.
        """
        if self.cache_theta is False:
            raise ValueError('cache_theta == False. Set ARTM.cache_theta = True')
        if not self._initialized:
            raise RuntimeError('Model does not exist yet. Use ARTM.initialize()/ARTM.fit_*()')

        theta_info = self.master.get_theta_info()

        column_names = []
        if self._theta_columns_naming == 'title':
            column_names = [item_title for item_title in theta_info.item_title]
        else:
            column_names = [item_id for item_id in theta_info.item_id]

        all_topic_names = [topic_name for topic_name in theta_info.topic_name]
        use_topic_names = topic_names if topic_names is not None else all_topic_names
        _, nd_array = self.master.get_theta_matrix(topic_names=use_topic_names)

        theta_data_frame = DataFrame(data=nd_array.transpose(),
                                     columns=column_names,
                                     index=use_topic_names)
        return theta_data_frame

    def remove_theta(self):
        """
        :Description: removes cached theta matrix
        """
        self.master.clear_theta_cache()

    def get_score(self, score_name):
        """
        :Description: get score after fit_offline, fit_online or transform

        :param str score_name: the name of the score to return
        """
        return self.master.get_score(score_name)

    def transform(self, batch_vectorizer=None, theta_matrix_type='dense_theta',
                  predict_class_id=None):
        """
        :Description: find Theta matrix for new documents

        :param object_reference batch_vectorizer: an instance of BatchVectorizer class
        :param str theta_matrix_type: type of matrix to be returned, possible values:
                'dense_theta', 'dense_ptdw', None, default='dense_theta'
        :param str predict_class_id: class_id of a target modality to predict.\
                When this option is enabled the resulting columns of theta matrix will\
                correspond to unique labels of a target modality. The values will represent\
                p(c|d), which give the probability of class label c for document d.

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the ids of documents, for which the Theta matrix was requested;
          * rows --- the names of topics in topic model, that was used to create Theta;
          * data --- content of Theta matrix.

        :Note:
          * 'dense_ptdw' mode provides simple access to values of p(t|w,d).
            The resulting pandas.DataFrame object will contain a flat theta matrix (no 3D) where
            each item has multiple columns - as many as the number of tokens in that document.
            These columns will have the same item_id.
            The order of columns with equal item_id is the same
            as the order of tokens in the input data (batch.item.token_id).
        """
        if batch_vectorizer is None:
            raise IOError('No batches were given for processing')

        if not self._initialized:
            raise RuntimeError('Model does not exist yet. Use ARTM.initialize()/ARTM.fit_*()')

        theta_matrix_type_real = const.ThetaMatrixType_None
        if theta_matrix_type == 'dense_theta':
            theta_matrix_type_real = const.ThetaMatrixType_Dense
        elif theta_matrix_type == 'sparse_theta':
            theta_matrix_type_real = const.ThetaMatrixType_Sparse
            raise NotImplementedError('Sparse format is currently unavailable from Python')
        elif theta_matrix_type == 'dense_ptdw':
            theta_matrix_type_real = const.ThetaMatrixType_DensePtdw
        elif theta_matrix_type == 'sparse_ptdw':
            theta_matrix_type_real = const.ThetaMatrixType_SparsePtdw
            raise NotImplementedError('Sparse format is currently unavailable from Python')

        batches_list = [batch.filename for batch in batch_vectorizer.batches_list]

        if theta_matrix_type is not None:
            theta_info, numpy_ndarray = self.master.transform(batch_filenames=batches_list,
                                                              theta_matrix_type=theta_matrix_type_real,
                                                              predict_class_id=predict_class_id)
            document_ids = []
            if self._theta_columns_naming == 'title':
                document_ids = [item_title for item_title in theta_info.item_title]
            else:
                document_ids = [item_id for item_id in theta_info.item_id]

            topic_names = [topic_name for topic_name in theta_info.topic_name]
            theta_data_frame = DataFrame(data=numpy_ndarray.transpose(),
                                         columns=document_ids,
                                         index=topic_names)
            return theta_data_frame
        else:
            self.master.transform(batch_filenames=batches_list,
                                  theta_matrix_type=theta_matrix_type_real,
                                  predict_class_id=predict_class_id)

    def initialize(self, dictionary=None):
        """
        :Description: initialize topic model before learning

        :param dictionary: loaded BigARTM collection dictionary
        :type dictionary: str or reference to Dictionary object
        """
        dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name

        self._lib.ArtmDisposeModel(self.master.master_id, self.model_pwt)
        self._lib.ArtmDisposeModel(self.master.master_id, self.model_nwt)
        self.master.initialize_model(model_name=self.model_pwt,
                                     dictionary_name=dictionary_name,
                                     topic_names=self._topic_names,
                                     seed=self._seed)
        self.master.initialize_model(model_name=self.model_nwt,
                                     dictionary_name=dictionary_name,
                                     topic_names=self._topic_names,
                                     seed=self._seed)

        topics_and_tokens_info = self.master.get_phi_info(self.model_pwt)

        self._topic_names = [topic_name for topic_name in topics_and_tokens_info.topic_name]
        self._initialized = True

        # Remove all info about previous iterations
        self._score_tracker = {}
        self._synchronizations_processed = 0
        self._num_online_processed_batches = 0
        self._phi_cached = None


def version():
    return ARTM(num_topics=1).library_version
