# Copyright 2017, Additive Regularization of Topic Models.

import datetime
import json
import os
import pickle
import uuid
import warnings

from copy import deepcopy
from multiprocessing.pool import (
    ApplyResult,
    ThreadPool,
)

import numpy
import tqdm

from packaging.version import parse
from pandas import DataFrame
from six import (
    iteritems,
    string_types,
)
from six.moves import (
    range,
    zip,
)

from . import master_component as mc
from . import score_tracker
from . import wrapper
from .batches_utils import BatchVectorizer
from .regularizers import Regularizers
from .scores import (
    Scores,
    TopicMassPhiScore,
)
from .wrapper import (
    constants as const,
    messages_pb2 as messages,
)

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

SCORE_TRACKER_FILENAME = 'score_tracker.bin'
PWT_FILENAME = 'p_wt.bin'
NWT_FILENAME = 'n_wt.bin'
PTD_FILENAME = 'p_td.bin'
PARAMETERS_FILENAME_JSON = 'parameters.json'
PARAMETERS_FILENAME_BIN = 'parameters.bin'


def _run_from_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except:  # noqa
        return False


def _topic_selection_regularizer_func(self, regularizers):
    topic_selection_regularizer_name = []
    for name, regularizer in iteritems(regularizers.data):
        if regularizer.type == const.RegularizerType_TopicSelectionTheta:
            topic_selection_regularizer_name.append(name)

    if len(topic_selection_regularizer_name):
        n_t = [0] * self.num_topics
        no_score = self._internal_topic_mass_score_name is None
        if no_score:
            self._internal_topic_mass_score_name = 'ITMScore_{}'.format(str(uuid.uuid4()))
            self.scores.add(TopicMassPhiScore(name=self._internal_topic_mass_score_name,
                                              model_name=self.model_nwt))

        if not self._synchronizations_processed or no_score:
            phi = self.get_phi()
            n_t = list(phi.sum(axis=0))
        else:
            last_topic_mass = self.score_tracker[self._internal_topic_mass_score_name].last_topic_mass
            for i, n in enumerate(self.topic_names):
                n_t[i] = last_topic_mass[n]

        n = sum(n_t)
        for name in topic_selection_regularizer_name:
            config = self.regularizers[name]._config_message()
            config.CopyFrom(self.regularizers[name].config)
            config.ClearField('topic_value')
            for value in [n / (e * self.num_topics) if e > 0.0 else 0.0 for e in n_t]:
                config.topic_value.append(value)
            self.regularizers[name].config = config


class ArtmThreadPool(object):
    def __init__(self, asynchronous=True):
        self._pool = ThreadPool(processes=1) if asynchronous else None

    def apply_async(self, func, args):
        return self._pool.apply_async(func, args) if self._pool else func(*args)

    def __deepcopy__(self, memo):
        return self


class ARTM(object):
    def __init__(self, num_topics=None, topic_names=None, num_processors=None, class_ids=None,
                 transaction_typenames=None, scores=None, regularizers=None, num_document_passes=10,
                 reuse_theta=False, dictionary=None, cache_theta=False, theta_columns_naming='id',
                 seed=-1, show_progress_bars=False, theta_name=None,
                 parent_model=None, parent_model_weight=None):
        """
        :param int num_topics: the number of topics in model, will be overwrited if\
                                 topic_names is set
        :param int num_processors: how many threads will be used for model training, if\
                                 not specified then number of threads will be detected by the lib
        :param topic_names: names of topics in model
        :type topic_names: list of str
        :param dict class_ids: list of class_ids and their weights to be used in model,\
                                 key --- class_id, value --- weight, if not specified then\
                                 all class_ids will be used.\
        :param dict transaction_typenames: list of transaction_typenames and their weights to be used in model,\
                                 key --- transaction_typename, value --- weight, if not specified then\
                                 all transaction_typenames will be used.\
                                 Specify class_ids parameter when using custom transaction_typenames parameter.
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
        :param show_progress_bars: a boolean flag indicating whether to show progress bar in fit_offline,\
                                   fit_online and transform operations.
        :type seed: unsigned int or -1
        :param theta_name: string, name of ptd (theta) matrix
        :param ARTM parent_model: An instance of ARTM class to use as parent level of hierarchy
        :param float parent_model_weight: weight of parent model (by default 1.0)

        :Important public fields:
          * regularizers: contains dict of regularizers, included into model
          * scores: contains dict of scores, included into model
          * score_tracker: contains dict of scoring results:\
               key --- score name, value --- ScoreTracker object, which contains info about\
               values of score on each synchronization (e.g. collection pass) in list

        :Note:
          * Here and anywhere in BigARTM empty topic_names, class_ids means that\
            model (or regularizer, or score) should use all topics and class_ids.\
            Don't confused with topic_name and class_id fields!
          * If some fields of regularizers or scores are not defined by\
            user --- internal lib defaults would be used.
          * If field 'topic_names' is None, it will be generated by BigARTM and will\
            be available using ARTM.topic_names().
          * Most arguments of ARTM constructor have corresponding setter and getter\
            of the same name that allows to change them at later time, after ARTM object\
            has been created.
          * Setting theta_name to a non-empty string activates an experimental mode\
            where cached theta matrix is internally stored as a phi matrix with tokens\
            corresponding to item title, so user should guarantee that all ites has unique titles.\
            With theta_name argument you specify the name of this matrix\
            (for example 'ptd' or 'theta', or whatever name you like).\
            Later you can retrieve this matix with ARTM.get_phi(model_name=ARTM.theta_name),\
            change its values with ARTM.master.attach_model(model=ARTM.theta_name),\
            export/import this matrix with ARTM.master.export_model('ptd', filename) and\
            ARTM.master.import_model('ptd', file_name). In this case you are also able to work\
            with theta matrix when using 'dump_artm_model' method and 'load_artm_model' function.
          * Setting parent_model parameter or, alternatively, calling ARTM.set_parent_model(),\
            cause this ARTM instance to behave as if it is a child level in hierarchical topic model.\
            This changes few things.\
            First, fit_offline() method will respect parent's model topics, as specified by\
            parent_model_weight paremeter. Larger values of parent_model_weight result in your\
            child model being more consistent with parent hierarchy. If you put parent_model_weight\
            as 0 your child level will be effectively independent from its parent.\
            Second, you may call ARTM.get_parent_psi() to retrieve a transition matrix, e.i. p(subtopic|topic).\
            Third, you no longer can use ARTM.fit_online(), which will throw an exception.\
            Fourth, you have to specify seed parameter (otherwise first topics in your child level will be initialized\
            the same way as in parent's model).\
            If you previously used hARTM class, this functionality is fully equivalent.\
            hARTM class is now deprecated. Note that dump_artm_model and load_artm_model is only partly supported.\
            After load_artm_model() you need to set parent model manually via set_parent_model(),\
            and also to specify value for ARTM.parent_model_weight property.
        """
        self._num_processors = None
        self._cache_theta = False
        self._parent_model_weight = None
        self._parent_model_id = None
        self._num_document_passes = num_document_passes
        self._reuse_theta = True
        self._theta_columns_naming = 'id'
        self._seed = -1
        self._show_progress_bars = show_progress_bars
        self._pool = ArtmThreadPool(asynchronous=show_progress_bars)

        if topic_names is not None:
            self._topic_names = topic_names
        elif num_topics is not None:
            self._topic_names = ['topic_{}'.format(i) for i in range(num_topics)]
        else:
            raise ValueError('Either num_topics or topic_names parameter should be set')

        self._class_ids = {}
        if class_ids is not None and isinstance(class_ids, dict) and len(class_ids) > 0:
            self._class_ids = class_ids

        self._transaction_typenames = {}
        if (transaction_typenames is not None and
                isinstance(transaction_typenames, dict) and len(transaction_typenames) > 0):  # noqa
            self._transaction_typenames = transaction_typenames

        if isinstance(num_processors, int) and num_processors > 0:
            self._num_processors = num_processors

        if isinstance(cache_theta, bool):
            self._cache_theta = cache_theta

        if isinstance(parent_model, ARTM):
            self._parent_model_id = parent_model.master.master_id

        if isinstance(parent_model_weight, (int, float)):
            self._parent_model_weight = parent_model_weight

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
        self._theta_name = theta_name

        self._lib = wrapper.LibArtm()
        master_config = messages.MasterModelConfig()
        if theta_name:
            master_config.ptd_name = theta_name
        self._master = mc.MasterComponent(self._lib,
                                          num_processors=self._num_processors,
                                          topic_names=self._topic_names,
                                          class_ids=self._class_ids,
                                          transaction_typenames=self._transaction_typenames,
                                          pwt_name=self._model_pwt,
                                          nwt_name=self._model_nwt,
                                          num_document_passes=self._num_document_passes,
                                          reuse_theta=self._reuse_theta,
                                          cache_theta=self._cache_theta,
                                          parent_model_id=self._parent_model_id,
                                          parent_model_weight=self._parent_model_weight,
                                          config=master_config)

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

    def clone(self):
        """
        :Description: returns a deep copy of the artm.ARTM object

        :Note:
          * This method is equivalent to copy.deepcopy() of your artm.ARTM object.
            Both methods perform deep copy of the object,
            including a complete copy of its internal C++ state
            (e.g. a copy of all phi and theta matrices, scores and regularizers,
            as well as ScoreTracker information with history of the scores).
          * Attached phi matrices are copied as dense phi matrices.
        """
        return deepcopy(self)

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
    def parent_model_weight(self):
        return self._parent_model_weight

    @property
    def theta_columns_naming(self):
        return self._theta_columns_naming

    @property
    def num_topics(self):
        return len(self._topic_names)

    @property
    def topic_names(self):
        """
        :Description: Gets or sets the list of topic names of the model.

        :Note:
          * Setting topic name allows you to put new labels on the existing topics.
            To add, remove or reorder topics use ARTM.reshape_topics() method.
          * In ARTM topic names are used just as string identifiers,
            which give a unique name to each column of the phi matrix.
            Typically you want to set topic names as something like "topic0", "topic1", etc.
            Later operations like get_phi() allow you to specify which topics you need to retrieve.
            Most regularizers allow you to limit the set of topics they act upon.
            If you configure a rich set of regularizers it is important design
            your topic names according to how they are regularizerd. For example,
            you may use names obj0, obj1, ..., objN for *objective* topics
            (those where you enable sparsity regularizers),
            and back0, back1, ..., backM for *background* topics
            (those where you enable smoothing regularizers).
        """
        return self._topic_names

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def transaction_typenames(self):
        return self._transaction_typenames

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
    def theta_name(self):
        return self._theta_name

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
    def show_progress_bars(self):
        return self._show_progress_bars

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

    @parent_model_weight.setter
    def parent_model_weight(self, parent_model_weight):
        if not isinstance(parent_model_weight, (int, float)):
            raise IOError('parent_model_weight should be float')
        else:
            self.master.reconfigure(parent_model_weight=parent_model_weight)
            self._parent_model_weight = parent_model_weight

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

    @transaction_typenames.setter
    def transaction_typenames(self, transaction_typenames):
        if len(transaction_typenames) < 0:
            raise IOError('Number of (transaction_typename, class_weight) pairs should be non-negative')
        else:
            self.master.reconfigure(transaction_typenames=transaction_typenames)
            self._transaction_typenames = transaction_typenames

    @seed.setter
    def seed(self, seed):
        if not isinstance(seed, int) or seed < 0:
            raise IOError('Random seed should be a positive integer')

        if self._initialized:
            # TODO: RuntimeError, not warning + test
            warnings.warn(
                message='Seed changing will have no effect:'
                        ' the model is already initialized!'
                        ' In the future versions of ARTM, this is going to be an Error.',
                category=RuntimeWarning,
            )

        self._seed = seed

    # ========== PRIVATE METHODS ==========
    def _wait_for_batches_processed(self, async_result, num_batches):
        if not(isinstance(async_result, ApplyResult)):
            return async_result

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            progress = tqdm.tqdm_notebook if _run_from_notebook() else tqdm.tqdm
            with progress(total=num_batches, desc='Batch', leave=False,
                          disable=not self._show_progress_bars) as batch_tqdm:
                previous_num_batches = 0
                while not async_result.ready():
                    async_result.wait(1)
                    current_num_batches = self.master.get_score(
                        score_name='^^^ItemsProcessedScore^^^').num_batches
                    batch_tqdm.update(current_num_batches - previous_num_batches)
                    previous_num_batches = current_num_batches
                return async_result.get()

    # ========== METHODS ==========
    def fit_offline(self, batch_vectorizer=None, num_collection_passes=1, reset_nwt=True):
        """
        :Description: proceeds the learning of topic model in offline mode

        :param object_referenece batch_vectorizer: an instance of BatchVectorizer class
        :param int num_collection_passes: number of iterations over whole given collection
        :param bool reset_nwt: a flag indicating whether to reset n_wt matrix to 0.
        """
        if batch_vectorizer is None:
            raise IOError('No batches were given for processing')

        if not self._initialized:
            raise RuntimeError('The model was not initialized. Use initialize() method')

        # outer cycle is needed because of TopicSelectionThetaRegularizer
        # and current ScoreTracker implementation

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            progress = tqdm.tnrange if _run_from_notebook() else tqdm.trange
            for _ in progress(num_collection_passes, desc='Pass',
                              disable=not self._show_progress_bars):
                # temp code for easy using of TopicSelectionThetaRegularizer from Python
                _topic_selection_regularizer_func(self, self._regularizers)

                self._synchronizations_processed += 1
                self._wait_for_batches_processed(
                    self._pool.apply_async(func=self.master.fit_offline,
                                           args=(batch_vectorizer.batches_ids,
                                                 batch_vectorizer.weights, 1, None, reset_nwt)),
                    batch_vectorizer.num_batches)

                for name in self.scores.data.keys():
                    if name not in self.score_tracker:
                        self.score_tracker[name] =\
                            SCORE_TRACKER[self.scores[name].type](self.scores[name])

        self._phi_cached = None

    def fit_online(self, batch_vectorizer=None, tau0=1024.0, kappa=0.7, update_every=1,
                   apply_weight=None, decay_weight=None, update_after=None, asynchronous=False):
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
        :param bool asynchronous: use or not the asynchronous implementation of the EM-algorithm

        :Note:
          asynchronous=True leads to impossibility of score extraction via score_tracker.\
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

        update_after_final, apply_weight_final, decay_weight_final = [], [], []
        if (update_after is None) or (apply_weight is None) or (decay_weight is None):
            update_after_final = range(update_every, batch_vectorizer.num_batches + 1, update_every)
            if len(update_after_final) == 0 or (update_after_final[-1] != batch_vectorizer.num_batches):
                update_after_final = list(update_after_final)
                update_after_final.append(batch_vectorizer.num_batches)

            for _ in update_after_final:
                self._num_online_processed_batches += update_every
                update_count = self._num_online_processed_batches // update_every
                rho = pow(tau0 + update_count, -kappa)
                apply_weight_final.append(rho)
                decay_weight_final.append(1 - rho)
        else:
            update_after_final = update_after
            apply_weight_final = apply_weight
            decay_weight_final = decay_weight

        # temp code for easy using of TopicSelectionThetaRegularizer from Python
        _topic_selection_regularizer_func(self, self._regularizers)

        self._wait_for_batches_processed(
            self._pool.apply_async(func=self.master.fit_online,
                                   args=(batch_vectorizer.batches_ids, batch_vectorizer.weights,
                                         update_after_final, apply_weight_final,
                                         decay_weight_final, asynchronous)),
            batch_vectorizer.num_batches)

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
          * Loaded model will overwrite ARTM.topic_names, class_ids and transaction_typenames fields.
          * All transaction_typenames (class_ids) weights will be set to 1.0, you need to specify them by\
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

        config = self._lib.ArtmRequestMasterModelConfig(self.master.master_id)
        self._topic_names = list(config.topic_name)

        class_ids = {}
        for class_id in config.class_id:
            class_ids[class_id] = 1.0
        self._class_ids = class_ids

        if hasattr(config, 'transaction_typename'):
            transaction_typenames = {}
            for transaction_typename in config.transaction_typename:
                transaction_typenames[transaction_typename] = 1.0
            self._transaction_typenames = transaction_typenames

        # Remove all info about previous iterations
        self._score_tracker = {}
        self._synchronizations_processed = 0
        self._num_online_processed_batches = 0
        self._phi_cached = None

    def get_phi_dense(self, topic_names=None, class_ids=None, model_name=None):
        """
        :Description: get phi matrix in dense format

        :param topic_names: list with topics or single topic to extract, None value means all topics
        :type topic_names: list of str or str or None
        :param class_ids: list with class_ids or single class_id to extract, None means all class ids
        :type class_ids: list of str or str or None
        :param str model_name: self.model_pwt by default, self.model_nwt is also\
                      reasonable to extract unnormalized counters

        :return:
          * a 3-tuple of (data, rows, columns), where
          * data --- numpy.ndarray with Phi data (i.e., p(w|t) values)
          * rows --- the tokens of topic model;
          * columns --- the names of topics in topic model;
        """
        if not self._initialized:
            raise RuntimeError('Model does not exist yet. Use ARTM.initialize()/ARTM.fit_*()')

        valid_model_name = self.model_pwt if model_name is None else model_name

        info = self.master.get_phi_info(valid_model_name)

        if isinstance(topic_names, string_types):
            topic_names = [topic_names]
        if isinstance(class_ids, string_types):
            class_ids = [class_ids]

        _, nd_array = self.master.get_phi_matrix(model=valid_model_name,
                                                 topic_names=topic_names,
                                                 class_ids=class_ids)

        tokens = [(c, t) for t, c in zip(info.token, info.class_id) if (class_ids is None or c in class_ids)]
        topic_names = [topic_name for topic_name in info.topic_name
                       if topic_names is None or topic_name in topic_names]
        return nd_array, tokens, topic_names

    def get_phi(self, topic_names=None, class_ids=None, model_name=None):
        """
        :Description: get custom Phi matrix of model. The extraction of the\
                      whole Phi matrix expects ARTM.phi_ call.

        :param topic_names: list with topics or single topic to extract, None value means all topics
        :type topic_names: list of str or str or None
        :param class_ids: list with class_ids or single class_id to extract, None means all class ids
        :type class_ids: list of str or str or None
        :param str model_name: self.model_pwt by default, self.model_nwt is also\
                      reasonable to extract unnormalized counters

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the names of topics in topic model;
          * rows --- the tokens of topic model;
          * data --- content of Phi matrix.
        """
        (nd_array, tokens, topic_names) = self.get_phi_dense(topic_names=topic_names,
                                                             class_ids=class_ids,
                                                             model_name=model_name)
        phi_data_frame = DataFrame(data=nd_array,
                                   columns=topic_names,
                                   index=tokens)

        return phi_data_frame

    def get_phi_sparse(self, topic_names=None, class_ids=None, model_name=None, eps=None):
        """
        :Description: get phi matrix in sparse format

        :param topic_names: list with topics or single topic to extract, None value means all topics
        :type topic_names: list of str or str or None
        :param class_ids: list with class_ids or single class_id to extract, None means all class ids
        :type class_ids: list of str or str or None
        :param str model_name: self.model_pwt by default, self.model_nwt is also\
                      reasonable to extract unnormalized counters
        :param float eps: threshold to consider values as zero

        :return:
          * a 3-tuple of (data, rows, columns), where
          * data --- scipy.sparse.csr_matrix with values
          * columns --- the names of topics in topic model;
          * rows --- the tokens of topic model;
        """
        from scipy import sparse

        if not self._initialized:
            raise RuntimeError('Model does not exist yet. Use ARTM.initialize()/ARTM.fit_*()')

        args = messages.GetTopicModelArgs()
        args.matrix_layout = wrapper.constants.MatrixLayout_Sparse
        if model_name is not None:
            args.model_name = model_name
        if eps is not None:
            args.eps = eps
        if topic_names is not None:
            if isinstance(topic_names, string_types):
                topic_names = [topic_names]
            for topic_name in topic_names:
                args.topic_name.append(topic_name)
        if class_ids is not None:
            if isinstance(class_ids, string_types):
                class_ids = [class_ids]
            for class_id in class_ids:
                args.class_id.append(class_id)

        tm = self._lib.ArtmRequestTopicModelExternal(self.master.master_id, args)

        numpy_ndarray = numpy.zeros(shape=(3 * tm.num_values, 1), dtype=numpy.float32)
        self._lib.ArtmCopyRequestedObject(numpy_ndarray)

        row_ind = numpy.frombuffer(numpy_ndarray.tobytes(), dtype=numpy.int32,
                                   count=tm.num_values, offset=0)
        col_ind = numpy.frombuffer(numpy_ndarray.tobytes(), dtype=numpy.int32,
                                   count=tm.num_values, offset=4*tm.num_values)
        data = numpy.frombuffer(numpy_ndarray.tobytes(), dtype=numpy.float32,
                                count=tm.num_values, offset=8*tm.num_values)

        # Rows correspond to tokens; get tokens from tm.token
        # Columns correspond to topics; get topic names from tm.topic_name
        data = sparse.csr_matrix((data, (row_ind, col_ind)),
                                 shape=(len(tm.token), len(tm.topic_name)))
        columns = list(tm.topic_name)

        rows = [(c, t) for (c, t) in zip(tm.token, tm.class_id)]
        return data, rows, columns

    def get_theta(self, topic_names=None):
        """
        :Description: get Theta matrix for training set of documents (or cached after transform)

        :param topic_names: list with topics or single topic to extract, None means all topics
        :type topic_names: list of str or str or None

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
        if isinstance(topic_names, string_types):
            topic_names = [topic_names]
        use_topic_names = topic_names if topic_names is not None else all_topic_names
        _, nd_array = self.master.get_theta_matrix(topic_names=use_topic_names)

        theta_data_frame = DataFrame(data=nd_array.transpose(),
                                     columns=column_names,
                                     index=use_topic_names)
        return theta_data_frame

    def get_theta_sparse(self, topic_names=None, eps=None):
        """
        :Description: get Theta matrix in sparse format

        :param topic_names: list with topics or single topic to extract, None means all topics
        :type topic_names: list of str or str or None
        :param float eps: threshold to consider values as zero

        :return:
          * a 3-tuple of (data, rows, columns), where
          * data --- scipy.sparse.csr_matrix with values
          * columns --- the ids of documents;
          * rows --- the names of topics in topic model;
        """
        from scipy import sparse

        if self.cache_theta is False:
            raise ValueError('cache_theta == False. Set ARTM.cache_theta = True')
        if not self._initialized:
            raise RuntimeError('Model does not exist yet. Use ARTM.initialize()/ARTM.fit_*()')

        args = messages.GetThetaMatrixArgs()
        args.matrix_layout = wrapper.constants.MatrixLayout_Sparse
        if eps is not None:
            args.eps = eps
        if topic_names is not None:
            if isinstance(topic_names, string_types):
                topic_names = [topic_names]
            for topic_name in topic_names:
                args.topic_name.append(topic_name)
        theta = self._lib.ArtmRequestThetaMatrixExternal(self.master.master_id, args)

        numpy_ndarray = numpy.zeros(shape=(3 * theta.num_values, 1), dtype=numpy.float32)
        self._lib.ArtmCopyRequestedObject(numpy_ndarray)

        col_ind = numpy.frombuffer(numpy_ndarray.tobytes(), dtype=numpy.int32,
                                   count=theta.num_values, offset=0)
        row_ind = numpy.frombuffer(numpy_ndarray.tobytes(), dtype=numpy.int32,
                                   count=theta.num_values, offset=4*theta.num_values)
        data = numpy.frombuffer(numpy_ndarray.tobytes(), dtype=numpy.float32,
                                count=theta.num_values, offset=8*theta.num_values)

        # Rows correspond to topics; get topic names from theta.topic_name
        # Columns correspond to items; get item IDs from theta.item_id
        data = sparse.csr_matrix((data, (row_ind, col_ind)),
                                 shape=(len(theta.topic_name), len(theta.item_id)))
        rows = list(theta.topic_name)
        columns = list(theta.item_title) if self._theta_columns_naming == 'title' else list(theta.item_id)
        return data, rows, columns

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

    def transform(self, batch_vectorizer=None, theta_matrix_type='dense_theta', predict_class_id=None):
        """
        :Description: find Theta matrix for new documents

        :param object_reference batch_vectorizer: an instance of BatchVectorizer class
        :param str theta_matrix_type: type of matrix to be returned, possible values:
                'dense_theta', 'dense_ptdw', 'cache', None, default='dense_theta'
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
        elif theta_matrix_type == 'cache':
            theta_matrix_type_real = const.ThetaMatrixType_Cache

        theta_info, numpy_ndarray = self._wait_for_batches_processed(
            self._pool.apply_async(func=self.master.transform,
                                   args=(None, batch_vectorizer.batches_ids, theta_matrix_type_real, predict_class_id)),
            batch_vectorizer.num_batches)

        if theta_matrix_type is not None and theta_matrix_type != 'cache':
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

    def transform_sparse(self, batch_vectorizer, eps=None):
        """
        :Description: find Theta matrix for new documents as sparse scipy matrix

        :param object_reference batch_vectorizer: an instance of BatchVectorizer class
        :param float eps: threshold to consider values as zero

        :return:
          * a 3-tuple of (data, rows, columns), where
          * data --- scipy.sparse.csr_matrix with values
          * columns --- the ids of documents;
          * rows --- the names of topics in topic model;
        """
        old_cache_theta = self.cache_theta
        self.cache_theta = True
        self.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type='cache')
        data, rows, columns = self.get_theta_sparse(eps=eps)
        self.cache_theta = old_cache_theta
        return data, rows, columns

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

        config = self._lib.ArtmRequestMasterModelConfig(self.master.master_id)
        self._topic_names = list(config.topic_name)
        self._initialized = True

        # Remove all info about previous iterations
        self._score_tracker = {}
        self._synchronizations_processed = 0
        self._num_online_processed_batches = 0
        self._phi_cached = None

    def reshape_topics(self, topic_names):
        """
        :Description: update topic names of the model.

        Adds, removes, or reorders columns of phi matrices
        according to the new set of topic names.
        New topics are initialized with zeros.
        """
        if not topic_names:
            raise IOError('Number of topic names should be non-negative')
        else:
            self.master.reconfigure_topic_name(topic_names=topic_names)
            self._topic_names = topic_names

    def reshape_tokens(self, dictionary):
        """
        :Description: update tokens of the model.

        Adds, removes, or reorders the tokens of the model
        according to a new dictionary.
        This operation changes n_wt matrix, but has no immediate effect on the p_wt matrix.
        You are expected to call ARTM.fit_offline() method
        to re-calculate p_wt matrix for the new set of tokens.
        """
        if not dictionary:
            raise IOError('Dictionary must not be None')
        dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
        self.master.initialize_model(model_name=self.model_nwt, dictionary_name=dictionary_name)

    def reshape(self, topic_names=None, dictionary=None):
        """
        :Description: change the shape of the model,
        e.i. add/remove topics, or add/remove tokens.

        :param topic_names: names of topics in model
        :type topic_names: list of str
        :param dictionary: dictionary that define new set of tokens
        :type dictionary: str or reference to Dictionary object

        Only one of the arguments (topic_names or dictionary) can be specified at a time.
        For further description see methods
        ARTM.reshape_topics() and ARTM.reshape_tokens().
        """
        if topic_names and dictionary:
            raise IOError('Only one of the arguments should be specified (topic_names or dictionary)')

        if topic_names:
            self.reshape_topics(topic_names)
            return

        if dictionary:
            self.reshape_tokens(dictionary)
            return

    def __repr__(self):
        if self.master is not None:
            num_tokens = next((x.num_tokens for x in self.info.model if x.name == self._model_pwt), None)
        else:
            num_tokens = "<UNKNOWN>"
        transaction_typenames = ', transaction_typenames={0}'.format(
            list(self.transaction_typenames.keys())) if self.transaction_typenames else ''
        return 'artm.ARTM(num_topics={0}, num_tokens={1}{2})'.format(
            self.num_topics, num_tokens, transaction_typenames)

    def dump_artm_model(self, data_path):
        """
        :Description: dump all necessary model files into given folder.

        :param str data_path: full path to folder (should unexist)
        """
        if os.path.exists(data_path):
            raise IOError('Folder {} already exists'.format(data_path))

        os.mkdir(data_path)
        # save core score tracker
        self._master.export_score_tracker(os.path.join(data_path, SCORE_TRACKER_FILENAME))
        # save phi and n_wt matrices
        self._master.export_model(self.model_pwt, os.path.join(data_path, PWT_FILENAME))
        self._master.export_model(self.model_nwt, os.path.join(data_path, NWT_FILENAME))
        # save theta if has theta_name
        if self.theta_name is not None:
            self._master.export_model(self.theta_name, os.path.join(data_path, PTD_FILENAME))

        # save parameters in human-readable format
        params = {}
        params['version'] = self.library_version
        params['creation_time'] = str(datetime.datetime.now())
        params['num_processors'] = self._num_processors
        params['cache_theta'] = self._cache_theta
        params['num_document_passes'] = self._num_document_passes
        params['reuse_theta'] = self._reuse_theta
        params['theta_columns_naming'] = self._theta_columns_naming
        params['seed'] = self._seed
        params['show_progress_bars'] = self._show_progress_bars
        params['topic_names'] = self._topic_names
        params['class_ids'] = self._class_ids
        params['transaction_typenames'] = self._transaction_typenames
        params['model_pwt'] = self._model_pwt
        params['model_nwt'] = self._model_nwt
        params['theta_name'] = self._theta_name
        params['synchronizations_processed'] = self._synchronizations_processed
        params['num_online_processed_batches'] = self._num_online_processed_batches
        params['initialized'] = self._initialized

        regularizers = {}
        for name, regularizer in iteritems(self._regularizers.data):
            tau = None
            gamma = None
            try:
                tau = regularizer.tau
                gamma = regularizer.gamma
            except KeyError:
                pass
            regularizers[name] = [str(regularizer.config), tau, gamma]
        params['regularizers'] = regularizers

        scores = {}
        for name, score in iteritems(self._scores.data):
            model_name = None
            try:
                model_name = score.model_name
            except KeyError:
                pass
            scores[name] = [str(score.config), model_name]

        params['scores'] = scores

        with open(os.path.join(data_path, PARAMETERS_FILENAME_JSON), 'w') as fout:
            json.dump(params, fout)

        # save parameters in binary format
        regularizers = {}
        for name, regularizer in iteritems(self._regularizers._data):
            regularizers[name] = [regularizer._config_message.__name__,
                                  regularizer.config.SerializeToString()]

            tau = None
            gamma = None
            try:
                tau = regularizer.tau
                gamma = regularizer.gamma
            except KeyError:
                pass

            if tau is not None:
                regularizers[name].append(tau)
                if gamma is not None:
                    regularizers[name].append(gamma)

        params['regularizers'] = regularizers

        scores = {}
        for name, score in iteritems(self._scores._data):
            scores[name] = [score._config_message.__name__,
                            score.config.SerializeToString()]

            model_name = None
            try:
                model_name = score.model_name
            except KeyError:
                pass
            if model_name is not None:
                scores[name].append(model_name)

        params['scores'] = scores

        with open(os.path.join(data_path, PARAMETERS_FILENAME_BIN), 'wb') as fout:
            pickle.dump(params, fout)

    def set_parent_model(self, parent_model, parent_model_weight=None):
        """
        :Description: sets parent model. For more details, see comment in ARTM.__init__.

        :param ARTM parent_model: An instance of ARTM class to use as parent level of hierarchy
        """

        if not isinstance(parent_model, ARTM):
            raise IOError('parent_model must be of type ARTM')

        self._parent_model_id = parent_model.master.master_id
        self.master.reconfigure(parent_model_id=self._parent_model_id)

        if parent_model_weight is not None:
            self.parent_model_weight = parent_model_weight

    def get_parent_psi(self):
        """
        :returns: p(subtopic|topic) matrix
        """
        if self._parent_model_id is None:
            raise IOError('get_parent_psi() require parent model to be set')

        batch = messages.Batch(id=str(uuid.uuid4()), description="__parent_phi_matrix_batch__")
        batch_vectorizer = BatchVectorizer(batches=[batch], process_in_memory_model=self)
        return self.transform(batch_vectorizer=batch_vectorizer)


def version():
    return ARTM(num_topics=1).library_version


def load_artm_model(data_path):
    """
    :Description: load all necessary files for model creation from given folder.

    :param str data_path: full path to folder (should exist)
    :return: artm.ARTM object, created using given dumped data
    """
    # load parameters
    with open(os.path.join(data_path, PARAMETERS_FILENAME_BIN), 'rb') as fin:
        params = pickle.load(fin)

    if parse(params['version']) > parse(version()):
        raise RuntimeError('File was generated with newer version of library ({}). '.format(params['version']) +
                           'Current library version is {}'.format(version()))

    model = ARTM(topic_names=params['topic_names'],
                 num_processors=params['num_processors'],
                 transaction_typenames=params['transaction_typenames'],
                 class_ids=params['class_ids'],
                 num_document_passes=params['num_document_passes'],
                 reuse_theta=params['reuse_theta'],
                 cache_theta=params['cache_theta'],
                 theta_columns_naming=params['theta_columns_naming'],
                 seed=params['seed'],
                 show_progress_bars=params['show_progress_bars'],
                 theta_name=params['theta_name'])

    model._model_pwt = params['model_pwt']
    model._model_nwt = params['model_nwt']
    model._synchronizations_processed = params['synchronizations_processed']
    model._num_online_processed_batches = params['num_online_processed_batches']
    model._initialized = params['initialized']

    for name, type_config in iteritems(params['regularizers']):
        config = None
        func = None
        for reg_info in mc.REGULARIZERS:
            if reg_info[0].__name__ == type_config[0]:
                config = reg_info[0]()
                func = reg_info[2]
        config.ParseFromString(type_config[1])

        if len(type_config) == 3:
            model.regularizers.add(func(name=name, config=config, tau=type_config[2]))
        elif len(type_config) == 4:
            model.regularizers.add(func(name=name, config=config, tau=type_config[2], gamma=type_config[3]))
        else:
            model.regularizers.add(func(name=name, config=config))

    # load scores and configure python score_tracker
    for name, type_config in iteritems(params['scores']):
        config = None
        func = None
        for score_info in mc.SCORES:
            if score_info[1].__name__ == type_config[0]:
                config = score_info[1]()
                func = score_info[3]
        config.ParseFromString(type_config[1])
        if len(type_config) == 3:
            model.scores.add(func(name=name, config=config, model_name=type_config[2]))
        else:
            model.scores.add(func(name=name, config=config))
        model.score_tracker[name] = SCORE_TRACKER[model.scores[name].type](model.scores[name])

    # load core score tracker
    model._master.import_score_tracker(os.path.join(data_path, SCORE_TRACKER_FILENAME))
    # load phi and n_wt matrices
    model._master.import_model(model.model_pwt, os.path.join(data_path, PWT_FILENAME))
    model._master.import_model(model.model_nwt, os.path.join(data_path, NWT_FILENAME))
    # load theta if has theta_name
    if model.theta_name is not None:
        model._master.import_model(model.theta_name, os.path.join(data_path, PTD_FILENAME))

    return model
