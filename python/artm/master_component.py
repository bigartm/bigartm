# Copyright 2017, Additive Regularization of Topic Models.

import os

import numpy

from six import iteritems
from six.moves import zip
import warnings

from . import regularizers
from . import scores
from .wrapper import (
    constants,
    messages_pb2 as messages,
)

REGULARIZERS = (
    (
        messages.SmoothSparseThetaConfig,
        constants.RegularizerType_SmoothSparseTheta,
        regularizers.SmoothSparseThetaRegularizer,
    ), (
        messages.SmoothSparsePhiConfig,
        constants.RegularizerType_SmoothSparsePhi,
        regularizers.SmoothSparsePhiRegularizer,
    ), (
        messages.DecorrelatorPhiConfig,
        constants.RegularizerType_DecorrelatorPhi,
        regularizers.DecorrelatorPhiRegularizer,
    ), (
        messages.LabelRegularizationPhiConfig,
        constants.RegularizerType_LabelRegularizationPhi,
        regularizers.LabelRegularizationPhiRegularizer,
    ), (
        messages.SpecifiedSparsePhiConfig,
        constants.RegularizerType_SpecifiedSparsePhi,
        regularizers.SpecifiedSparsePhiRegularizer,
    ), (
        messages.ImproveCoherencePhiConfig,
        constants.RegularizerType_ImproveCoherencePhi,
        regularizers.ImproveCoherencePhiRegularizer,
    ), (
        messages.SmoothPtdwConfig,
        constants.RegularizerType_SmoothPtdw,
        regularizers.SmoothPtdwRegularizer,
    ), (
        messages.TopicSelectionThetaConfig,
        constants.RegularizerType_TopicSelectionTheta,
        regularizers.TopicSelectionThetaRegularizer,
    ), (
        messages.BitermsPhiConfig,
        constants.RegularizerType_BitermsPhi,
        regularizers.BitermsPhiRegularizer,
    ), (
        messages.HierarchySparsingThetaConfig,
        constants.RegularizerType_HierarchySparsingTheta,
        regularizers.HierarchySparsingThetaRegularizer,
    ), (
        messages.TopicSegmentationPtdwConfig,
        constants.RegularizerType_TopicSegmentationPtdw,
        regularizers.TopicSegmentationPtdwRegularizer,
    ), (
        messages.SmoothTimeInTopicsPhiConfig,
        constants.RegularizerType_SmoothTimeInTopicsPhi,
        regularizers.SmoothTimeInTopicsPhiRegularizer,
    ), (
        messages.NetPlsaPhiConfig,
        constants.RegularizerType_NetPlsaPhi,
        regularizers.NetPlsaPhiRegularizer,
    ),
)

SCORES = (
    (
        constants.ScoreType_Perplexity,
        messages.PerplexityScoreConfig,
        messages.PerplexityScore,
        scores.PerplexityScore,
    ), (
        constants.ScoreType_SparsityTheta,
        messages.SparsityThetaScoreConfig,
        messages.SparsityThetaScore,
        scores.SparsityThetaScore,
    ), (
        constants.ScoreType_SparsityPhi,
        messages.SparsityPhiScoreConfig,
        messages.SparsityPhiScore,
        scores.SparsityPhiScore,
    ), (
        constants.ScoreType_ItemsProcessed,
        messages.ItemsProcessedScoreConfig,
        messages.ItemsProcessedScore,
        scores.ItemsProcessedScore,
    ), (
        constants.ScoreType_TopTokens,
        messages.TopTokensScoreConfig,
        messages.TopTokensScore,
        scores.TopTokensScore,
    ), (
        constants.ScoreType_ThetaSnippet,
        messages.ThetaSnippetScoreConfig,
        messages.ThetaSnippetScore,
        scores.ThetaSnippetScore,
    ), (
        constants.ScoreType_TopicKernel,
        messages.TopicKernelScoreConfig,
        messages.TopicKernelScore,
        scores.TopicKernelScore,
    ), (
        constants.ScoreType_TopicMassPhi,
        messages.TopicMassPhiScoreConfig,
        messages.TopicMassPhiScore,
        scores.TopicMassPhiScore,

    ), (
        constants.ScoreType_ClassPrecision,
        messages.ClassPrecisionScoreConfig,
        messages.ClassPrecisionScore,
        scores.ClassPrecisionScore,
    ), (
        constants.ScoreType_BackgroundTokensRatio,
        messages.BackgroundTokensRatioScoreConfig,
        messages.BackgroundTokensRatioScore,
        scores.BackgroundTokensRatioScore,
    ),
)


def _regularizer_type(config):
    for mcls, const, _ in REGULARIZERS:
        if isinstance(config, mcls):
            return const
    warnings.warn(
        message='Failed to determine the config type of {config}.'
                ' Make sure that all regularizers are defined correctly'
    )


def _score_type(config):
    for const, ccls, _, _ in SCORES:
        if isinstance(config, ccls):
            return const
    warnings.warn(
        message='Failed to determine the config type of {config}.'
                ' Make sure that all scores are defined correctly'
    )


def _score_data_func(score_data_type):
    for const, _, mfunc, _ in SCORES:
        if score_data_type == const:
            return mfunc
    warnings.warn(
        message='Failed to determine the config type of {config}.'
                ' Make sure that all scores are defined correctly'
    )


def _prepare_config(topic_names=None, class_ids=None, transaction_typenames=None,
                    scores=None, regularizers=None, num_processors=None,
                    pwt_name=None, nwt_name=None, num_document_passes=None,
                    reuse_theta=None, cache_theta=None,
                    parent_model_id=None, parent_model_weight=None, args=None):
        master_config = messages.MasterModelConfig()

        if args is not None:
            master_config.CopyFrom(args)

        if topic_names is not None:
            master_config.ClearField('topic_name')
            for topic_name in topic_names:
                master_config.topic_name.append(topic_name)

        if transaction_typenames is not None:
            master_config.ClearField('transaction_typename')
            master_config.ClearField('transaction_weight')
            for transaction_typename, transaction_weight in iteritems(transaction_typenames):
                master_config.transaction_typename.append(transaction_typename)
                master_config.transaction_weight.append(transaction_weight)

        if class_ids is not None:
            master_config.ClearField('class_id')
            master_config.ClearField('class_weight')
            for class_id, class_weight in iteritems(class_ids):
                master_config.class_id.append(class_id)
                master_config.class_weight.append(class_weight)

        if scores is not None:
            master_config.ClearField('score_config')
            for name, config in iteritems(scores):
                score_config = master_config.score_config.add()
                score_config.name = name
                score_config.type = _score_type(config)
                score_config.config = config.SerializeToString()

        if regularizers is not None:
            master_config.ClearField('regularizer_config')
            for name, config_tau_gamma in iteritems(regularizers):
                regularizer_config = master_config.regularizer_config.add()
                regularizer_config.name = name
                regularizer_config.type = _regularizer_type(config)
                regularizer_config.config = config_tau_gamma[0].SerializeToString()
                regularizer_config.tau = config_tau_gamma[1]
                if len(config_tau_gamma) == 3 and config_tau_gamma[2] is not None:
                    regularizer_config.gamma = config_tau_gamma[2]

        if num_processors is not None:
            master_config.num_processors = num_processors

        if reuse_theta is not None:
            master_config.reuse_theta = reuse_theta

        if cache_theta is not None:
            master_config.cache_theta = cache_theta

        if parent_model_id is not None:
            master_config.parent_master_model_id = parent_model_id
            master_config.opt_for_avx = False

        if parent_model_weight is not None:
            master_config.parent_master_model_weight = parent_model_weight

        if pwt_name is not None:
            master_config.pwt_name = pwt_name

        if nwt_name is not None:
            master_config.nwt_name = nwt_name

        if num_document_passes is not None:
            master_config.num_document_passes = num_document_passes

        return master_config


class MasterComponent(object):
    def __init__(self, library=None, topic_names=None, class_ids=None, transaction_typenames=None,
                 scores=None, regularizers=None, num_processors=None, pwt_name=None,
                 nwt_name=None, num_document_passes=None, reuse_theta=None,
                 cache_theta=False, parent_model_id=None, parent_model_weight=None,
                 config=None, master_id=None):
        """

        :param library: an instance of LibArtm
        :param topic_names: list of topic names to use in model
        :type topic_names: list of str
        :param dict class_ids: key - class_id, value - class_weight
        :param dict transaction_typenames: key - transaction_typename, value - transaction_weight,\
                                       specify class_ids when using custom transaction_typenames
        :param dict scores: key - score name, value - config
        :param dict regularizers: key - regularizer name, value - tuple (config, tau)\
                                  or triple (config, tau, gamma)
        :param int num_processors: number of worker threads to use for processing the collection
        :param str pwt_name: name of pwt matrix
        :param str nwt_name: name of nwt matrix
        :param in num_document_passes: num passes through each document
        :param bool reuse_theta: reuse Theta from previous iteration or not
        :param bool cache_theta: save or not the Theta matrix
        :param int parent_model_id: master_id of parent model (previous level of hierarchy)
        :param float parent_model_weight: weight of parent model (plays role in fit_offline;
                                          defines how much to respect parent model as compared to batches)
        """
        self._lib = library

        master_config = _prepare_config(topic_names=topic_names,
                                        class_ids=class_ids,
                                        transaction_typenames=transaction_typenames,
                                        scores=scores,
                                        regularizers=regularizers,
                                        num_processors=num_processors,
                                        pwt_name=pwt_name,
                                        nwt_name=nwt_name,
                                        num_document_passes=num_document_passes,
                                        reuse_theta=reuse_theta,
                                        cache_theta=cache_theta,
                                        parent_model_id=parent_model_id,
                                        parent_model_weight=parent_model_weight,
                                        args=config)

        self._config = master_config
        self.master_id = master_id if master_id is not None else self._lib.ArtmCreateMasterModel(master_config)

    def __deepcopy__(self, memo):
        new_master_id = self._lib.ArtmDuplicateMasterComponent(
            self.master_id, messages.DuplicateMasterComponentArgs())
        return MasterComponent(self._lib, config=self._config, master_id=new_master_id)

    def reconfigure(self, topic_names=None, class_ids=None, transaction_typenames=None,
                    scores=None, regularizers=None, num_processors=None, pwt_name=None,
                    nwt_name=None, num_document_passes=None, reuse_theta=None, cache_theta=None,
                    parent_model_id=None, parent_model_weight=None):
        master_config = _prepare_config(topic_names=topic_names,
                                        class_ids=class_ids,
                                        transaction_typenames=transaction_typenames,
                                        scores=scores,
                                        regularizers=regularizers,
                                        num_processors=num_processors,
                                        pwt_name=pwt_name,
                                        nwt_name=nwt_name,
                                        num_document_passes=num_document_passes,
                                        reuse_theta=reuse_theta,
                                        cache_theta=cache_theta,
                                        parent_model_id=parent_model_id,
                                        parent_model_weight=parent_model_weight,
                                        args=self._config)

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def reconfigure_topic_name(self, topic_names=None):
        master_config = _prepare_config(topic_names=topic_names,
                                        args=self._config)

        self._config = master_config
        self._lib.ArtmReconfigureTopicName(self.master_id, master_config)

    def import_dictionary(self, filename, dictionary_name):
        """
        :param str filename: full name of dictionary file
        :param str dictionary_name: name of imported dictionary
        """
        args = messages.ImportDictionaryArgs(dictionary_name=dictionary_name, file_name=filename)
        self._lib.ArtmImportDictionary(self.master_id, args)

    def export_dictionary(self, filename, dictionary_name):
        """
        :param str filename: full name of dictionary file
        :param str dictionary_name: name of exported dictionary
        """
        args = messages.ExportDictionaryArgs(dictionary_name=dictionary_name, file_name=filename)
        self._lib.ArtmExportDictionary(self.master_id, args)

    def create_dictionary(self, dictionary_data, dictionary_name=None):
        """
        :param dictionary_data: an instance of DictionaryData with info about dictionary
        :param str dictionary_name: name of exported dictionary
        """

        if dictionary_name is not None:
            dictionary_data.name = dictionary_name

        self._lib.ArtmCreateDictionary(self.master_id, dictionary_data)

    def get_dictionary(self, dictionary_name):
        """
        :param str dictionary_name: name of dictionary to get
        """
        args = messages.GetDictionaryArgs(dictionary_name=dictionary_name)
        dictionary_data = self._lib.ArtmRequestDictionary(self.master_id, args)
        return dictionary_data

    def gather_dictionary(self, dictionary_target_name=None, data_path=None, cooc_file_path=None,
                          vocab_file_path=None, symmetric_cooc_values=None, args=None):
        """
        :param str dictionary_target_name: name of the dictionary in the core
        :param str data_path: full path to batches folder
        :param str cooc_file_path: full path to the file with cooc info
        :param str vocab_file_path: full path to the file with vocabulary
        :param bool symmetric_cooc_values: whether the cooc matrix should\
                considered to be symmetric or not
        :param args: an instance of GatherDictionaryArgs
        """
        gather_args = messages.GatherDictionaryArgs()
        if args is not None:
            gather_args = args
        if dictionary_target_name is not None:
            gather_args.dictionary_target_name = dictionary_target_name
        if data_path is not None:
            gather_args.data_path = data_path
        if cooc_file_path is not None:
            gather_args.cooc_file_path = cooc_file_path
        if vocab_file_path is not None:
            gather_args.vocab_file_path = vocab_file_path
        if symmetric_cooc_values is not None:
            gather_args.symmetric_cooc_values = symmetric_cooc_values

        self._lib.ArtmGatherDictionary(self.master_id, gather_args)

    def filter_dictionary(self, dictionary_name=None, dictionary_target_name=None, class_id=None,
                          min_df=None, max_df=None,
                          min_df_rate=None, max_df_rate=None,
                          min_tf=None, max_tf=None,
                          max_dictionary_size=None,
                          recalculate_value=None,
                          args=None):

        """
        :param str dictionary_name: name of the dictionary in the core to filter
        :param str dictionary_target_name: name for the new filtered dictionary in the core
        :param str class_id: class_id to filter
        :param float min_df: min df value to pass the filter
        :param float max_df: max df value to pass the filter
        :param float min_df_rate: min df rate to pass the filter
        :param float max_df_rate: max df rate to pass the filter
        :param float min_tf: min tf value to pass the filter
        :param float max_tf: max tf value to pass the filter
        :param int max_dictionary_size: give an easy option to limit dictionary size;\
            rare tokens will be excluded until dictionary reaches given size
        :param bool recalculate_value: recalculate or not value field in dictionary\
            after filtration according to new sum of tf values
        :param args: an instance of FilterDictionaryArgs
        """
        filter_args = messages.FilterDictionaryArgs()
        if args is not None:
            filter_args = args
        if dictionary_target_name is not None:
            filter_args.dictionary_target_name = dictionary_target_name
        if dictionary_name is not None:
            filter_args.dictionary_name = dictionary_name
        if class_id is not None:
            filter_args.class_id = class_id
        if min_df is not None:
            filter_args.min_df = min_df
        if max_df is not None:
            filter_args.max_df = max_df
        if min_df_rate is not None:
            filter_args.min_df_rate = min_df_rate
        if max_df_rate is not None:
            filter_args.max_df_rate = max_df_rate
        if min_tf is not None:
            filter_args.min_tf = min_tf
        if max_tf is not None:
            filter_args.max_tf = max_tf
        if max_dictionary_size is not None:
            filter_args.max_dictionary_size = max_dictionary_size
        if recalculate_value is not None:
            filter_args.recalculate_value = recalculate_value

        self._lib.ArtmFilterDictionary(self.master_id, filter_args)

    def initialize_model(self, model_name=None, topic_names=None,
                         dictionary_name=None, seed=None, args=None):
        """
        :param str model_name: name of pwt matrix in BigARTM
        :param topic_names: the list of names of topics to be used in model
        :type topic_names: list of str
        :param str dictionary_name: name of imported dictionary
        :param seed: seed for random initialization, None means no seed
        :type seed: unsigned int or -1, default None
        :param args: an instance of InitilaizeModelArgs
        """
        init_args = messages.InitializeModelArgs()
        if args is not None:
            init_args = args
        if model_name is not None:
            init_args.model_name = model_name
        if seed is not None:
            init_args.seed = seed
        if topic_names is not None:
            init_args.ClearField('topic_name')
            for topic_name in topic_names:
                init_args.topic_name.append(topic_name)

        init_args.dictionary_name = dictionary_name
        self._lib.ArtmInitializeModel(self.master_id, init_args)

    def clear_theta_cache(self):
        """
        Clears all entries from theta matrix cache
        """
        args = messages.ClearThetaCacheArgs()
        self._lib.ArtmClearThetaCache(self.master_id, args)

    def clear_score_cache(self):
        """
        Clears all entries from score cache
        """
        args = messages.ClearScoreCacheArgs()
        self._lib.ArtmClearScoreCache(self.master_id, args)

    def clear_score_array_cache(self):
        """
        Clears all entries from score array cache
        """
        args = messages.ClearScoreArrayCacheArgs()
        self._lib.ArtmClearScoreArrayCache(self.master_id, args)

    def process_batches(self, pwt, nwt=None, num_document_passes=None, batches_folder=None,
                        batches=None, regularizer_name=None, regularizer_tau=None,
                        class_ids=None, class_weights=None, find_theta=False,
                        transaction_typenames=None, transaction_weights=None,
                        reuse_theta=False, find_ptdw=False,
                        predict_class_id=None, predict_transaction_type=None):
        """
        :param str pwt: name of pwt matrix in BigARTM
        :param str nwt: name of nwt matrix in BigARTM
        :param int num_document_passes: number of inner iterations during processing
        :param str batches_folder: full path to data folder (alternative 1)
        :param batches: full file names of batches to process (alternative 2)
        :type batches: list of str
        :param regularizer_name: list of names of Theta regularizers to use
        :type regularizer_name: list of str
        :param regularizer_tau: list of tau coefficients for Theta regularizers
        :type regularizer_tau: list of float
        :param class_ids: list of class ids to use during processing.
        :type class_ids: list of str
        :param class_weights: list of corresponding weights of class ids.
        :type class_weights: list of float
        :param transaction_typenames: list of transaction types to use during processing.
        :type transaction_typenames: list of str
        :param transaction_weights: list of corresponding weights of transaction types.
        :type transaction_weights: list of float
        :param bool find_theta: find theta matrix for 'batches' (if alternative 2)
        :param bool reuse_theta: initialize by theta from previous collection pass
        :param bool find_ptdw: calculate and return Ptdw matrix or not\
                (works if find_theta == False)
        :param predict_class_id: class_id of a target modality to predict
        :type predict_class_id: str, default None
        :return:
            * tuple (messages.ThetaMatrix, numpy.ndarray) --- the info about Theta\
                    (if find_theta == True)
            * messages.ThetaMatrix --- the info about Theta (if find_theta == False)
        """
        args = messages.ProcessBatchesArgs(pwt_source_name=pwt,
                                           reuse_theta=reuse_theta)
        if nwt is not None:
            args.nwt_target_name = nwt
        if batches_folder is not None:
            for name in os.listdir(batches_folder):
                _, extension = os.path.splitext(name)
                if extension == '.batch':
                    args.batch_filename.append(os.path.join(batches_folder, name))
        if batches is not None:
            for batch in batches:
                args.batch_filename.append(batch)

        if num_document_passes is not None:
            args.num_document_passes = num_document_passes

        if regularizer_name is not None and regularizer_tau is not None:
            for name, tau in zip(regularizer_name, regularizer_tau):
                args.regularizer_name.append(name)
                args.regularizer_tau.append(tau)

        if transaction_typenames is not None and transaction_weights is not None:
            for transaction_typename, weight in zip(transaction_typenames, transaction_weights):
                args.transaction_typename.append(transaction_typename)
                args.transaction_weight.append(weight)

        if class_ids is not None and class_weights is not None:
            for class_id, weight in zip(class_ids, class_weights):
                args.class_id.append(class_id)
                args.class_weight.append(weight)

        if predict_class_id is not None:
            args.predict_class_id = predict_class_id

        func = None
        if find_theta or find_ptdw:
            args.theta_matrix_type = constants.ThetaMatrixType_Dense
            if find_ptdw:
                args.theta_matrix_type = constants.ThetaMatrixType_DensePtdw
            func = self._lib.ArtmRequestProcessBatchesExternal
        elif not find_theta or find_theta is None:
            func = self._lib.ArtmRequestProcessBatches

        result = func(self.master_id, args)

        if not find_theta and not find_ptdw:
            return result.theta_matrix

        num_rows = len(result.theta_matrix.item_id)
        num_cols = result.theta_matrix.num_topics
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)
        self._lib.ArtmCopyRequestedObject(numpy_ndarray)

        return result.theta_matrix, numpy_ndarray

    def regularize_model(self, pwt, nwt, rwt, regularizer_name, regularizer_tau, regularizer_gamma=None):
        """
        :param str pwt: name of pwt matrix in BigARTM
        :param str nwt: name of nwt matrix in BigARTM
        :param str rwt: name of rwt matrix in BigARTM
        :param regularizer_name: list of names of Phi regularizers to use
        :type regularizer_name: list of str
        :param regularizer_tau: list of tau coefficients for Phi regularizers
        :type regularizer_tau: list of floats
        :type regularizer_tau: list of floats
        """
        args = messages.RegularizeModelArgs(pwt_source_name=pwt,
                                            nwt_source_name=nwt,
                                            rwt_target_name=rwt)

        _gamma = regularizer_gamma
        if regularizer_gamma is None:
            _gamma = [None] * len(regularizer_tau)

        for name, tau, gamma in zip(regularizer_name, regularizer_tau, _gamma):
            reg_set = args.regularizer_settings.add()
            reg_set.name = name
            reg_set.tau = tau
            if gamma is not None:
                reg_set.gamma = gamma

        self._lib.ArtmRegularizeModel(self.master_id, args)

    def normalize_model(self, pwt, nwt, rwt=None):
        """
        :param str pwt: name of pwt matrix in BigARTM
        :param str nwt: name of nwt matrix in BigARTM
        :param str rwt: name of rwt matrix in BigARTM
        """
        args = messages.NormalizeModelArgs(pwt_target_name=pwt, nwt_source_name=nwt)
        if rwt is not None:
            args.rwt_source_name = rwt

        self._lib.ArtmNormalizeModel(self.master_id, args)

    def merge_model(self, models, nwt, topic_names=None, dictionary_name=None):
        """
        Merge multiple nwt-increments together.

        :param dict models: list of models with nwt-increments and their weights,\
                key - nwt_source_name, value - source_weight.
        :param str nwt: the name of target matrix to store combined nwt.\
                The matrix will be created by this operation.
        :param topic_names: names of topics in the resulting model. By default model\
                names are taken from the first model in the list.
        :param dictionary_name: name of dictionary that defines which tokens to include in merged model
        :type topic_names: list of str
        """
        args = messages.MergeModelArgs(nwt_target_name=nwt)
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)
        if dictionary_name is not None:
            args.dictionary_name = dictionary_name
        for nwt_source_name, source_weight in iteritems(models):
            args.nwt_source_name.append(nwt_source_name)
            args.source_weight.append(source_weight)

        self._lib.ArtmMergeModel(self.master_id, args)

    def attach_model(self, model):
        """
        :param str model: name of matrix in BigARTM
        :return:
            * messages.TopicModel() object with info about Phi matrix
            * numpy.ndarray with Phi data (i.e., p(w|t) values)
        """
        topic_model = self.get_phi_info(model)

        num_rows = len(topic_model.token)
        num_cols = topic_model.num_topics
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)

        self._lib.ArtmAttachModel(self.master_id,
                                  messages.AttachModelArgs(model_name=model),
                                  numpy_ndarray)

        return topic_model, numpy_ndarray

    def create_regularizer(self, name, config, tau, gamma=None):
        """
        :param str name: the name of the future regularizer
        :param config: the config of the future regularizer
        :param float tau: the coefficient of the regularization
        """
        master_config = messages.MasterModelConfig()
        master_config.CopyFrom(self._config)

        regularizer_config = master_config.regularizer_config.add()
        regularizer_config.name = name
        regularizer_config.type = _regularizer_type(config)
        regularizer_config.config = config.SerializeToString()
        regularizer_config.tau = tau
        if gamma is not None:
            regularizer_config.gamma = gamma

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def reconfigure_regularizer(self, name, config=None, tau=None, gamma=None):
        master_config = messages.MasterModelConfig()
        master_config.CopyFrom(self._config)

        for index, regularizer_config in enumerate(master_config.regularizer_config):
            if regularizer_config.name == name:
                if config is not None:
                    master_config.regularizer_config[index].config = config.SerializeToString()
                if tau is not None:
                    master_config.regularizer_config[index].tau = tau
                if gamma is not None:
                    master_config.regularizer_config[index].gamma = gamma

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def create_score(self, name, config, model_name=None):
        """
        :param str name: the name of the future score
        :param config: an instance of \*\*\*ScoreConfig
        :param model_name: pwt or nwt model name
        """
        master_config = messages.MasterModelConfig()
        master_config.CopyFrom(self._config)

        score_config = master_config.score_config.add()
        score_config.name = name
        score_config.type = _score_type(config)
        score_config.config = config.SerializeToString()

        if model_name is not None:
            score_config.model_name = model_name

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def get_score(self, score_name):
        """
        :param str score_name: the user defined name of score to retrieve
        :param score_config: reference to score data object
        """
        args = messages.GetScoreValueArgs(score_name=score_name)
        score_data = self._lib.ArtmRequestScore(self.master_id, args)

        score_info = _score_data_func(score_data.type)()
        score_info.ParseFromString(score_data.data)

        return score_info

    def get_score_array(self, score_name):
        """
        :param str score_name: the user defined name of score to retrieve
        :param score_config: reference to score data object
        """
        args = messages.GetScoreArrayArgs(score_name=score_name)
        score_array = self._lib.ArtmRequestScoreArray(self.master_id, args)

        scores = []
        for score_data in score_array.score:
            score_info = _score_data_func(score_data.type)()
            score_info.ParseFromString(score_data.data)
            scores.append(score_info)

        return scores

    def reconfigure_score(self, name, config, model_name=None):
        master_config = messages.MasterModelConfig()
        master_config.CopyFrom(self._config)

        for index, score_config in enumerate(master_config.score_config):
            if score_config.name == name:
                master_config.score_config[index].config = config.SerializeToString()
            if model_name is not None:
                score_config.model_name = model_name

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def get_theta_info(self):
        """
        :return: messages.ThetaMatrix object
        """
        args = messages.GetThetaMatrixArgs()
        args.matrix_layout = constants.MatrixLayout_Sparse
        args.eps = 1.001  # hack to not get any data back
        theta_matrix_info = self._lib.ArtmRequestThetaMatrix(self.master_id, args)

        return theta_matrix_info

    def get_theta_matrix(self, topic_names=None):
        """
        :param topic_names: list of topics to retrieve (None means all topics)
        :type topic_names: list of str or None
        :return: numpy.ndarray with Theta data (i.e., p(t|d) values)
        """
        args = messages.GetThetaMatrixArgs()
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)

        theta_matrix_info = self._lib.ArtmRequestThetaMatrixExternal(self.master_id, args)

        num_rows = len(theta_matrix_info.item_id)
        num_cols = theta_matrix_info.num_topics
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)
        self._lib.ArtmCopyRequestedObject(numpy_ndarray)

        return theta_matrix_info, numpy_ndarray

    def get_phi_info(self, model):
        """
        :param str model: name of matrix in BigARTM
        :return: messages.TopicModel object
        """
        args = messages.GetTopicModelArgs(model_name=model)
        args.matrix_layout = constants.MatrixLayout_Sparse
        args.eps = 1.001  # hack to not get any data back
        phi_matrix_info = self._lib.ArtmRequestTopicModel(self.master_id, args)

        return phi_matrix_info

    def get_phi_matrix(self, model, topic_names=None, class_ids=None, use_sparse_format=None):
        """
        :param str model: name of matrix in BigARTM
        :param topic_names: list of topics to retrieve (None means all topics)
        :type topic_names: list of str or None
        :param class_ids: list of class ids to retrieve (None means all class ids)
        :type class_ids: list of str or None
        :param bool use_sparse_format: use sparse\dense layout
        :return: numpy.ndarray with Phi data (i.e., p(w|t) values)
        """
        args = messages.GetTopicModelArgs(model_name=model)
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)
        if class_ids is not None:
            args.ClearField('class_id')
            for class_id in class_ids:
                args.class_id.append(class_id)
        if use_sparse_format is not None:
            args.matrix_layout = constants.MatrixLayout_Sparse

        phi_matrix_info = self._lib.ArtmRequestTopicModelExternal(self.master_id, args)

        num_rows = len(phi_matrix_info.token)
        num_cols = phi_matrix_info.num_topics
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)
        self._lib.ArtmCopyRequestedObject(numpy_ndarray)

        return phi_matrix_info, numpy_ndarray

    def export_model(self, model, filename):
        """
        :param str model: name of matrix in BigARTM
        :param str filename: the name of file to save model into binary format
        """
        args = messages.ExportModelArgs(model_name=model, file_name=filename)
        result = self._lib.ArtmExportModel(self.master_id, args)

    def import_model(self, model, filename):
        """
        :param str model: name of matrix in BigARTM
        :param str filename: the name of file to load model from binary format
        """
        args = messages.ImportModelArgs(model_name=model, file_name=filename)
        result = self._lib.ArtmImportModel(self.master_id, args)

    def get_info(self):
        info = self._lib.ArtmRequestMasterComponentInfo(self.master_id,
                                                        messages.GetMasterComponentInfoArgs())
        return info

    def fit_offline(self, batch_filenames=None, batch_weights=None,
                    num_collection_passes=None, batches_folder=None,
                    reset_nwt=True):
        """
        :param batch_filenames: name of batches to process
        :type batch_filenames: list of str
        :param batch_weights: weights of batches to process
        :type batch_weights: list of float
        :param int num_collection_passes: number of outer iterations
        :param str batches_folder: folder containing batches to process
        :param bool reset_nwt: a flag indicating whether to reset n_wt matrix to 0.
        """
        args = messages.FitOfflineMasterModelArgs()
        args.reset_nwt = reset_nwt
        if batch_filenames is not None:
            args.ClearField('batch_filename')
            for filename in batch_filenames:
                args.batch_filename.append(filename)

        if batch_weights is not None:
            args.ClearField('batch_weight')
            for weight in batch_weights:
                args.batch_weight.append(weight)

        if num_collection_passes is not None:
            args.num_collection_passes = num_collection_passes

        if batches_folder is not None:
            args.batch_folder = batches_folder

        self._lib.ArtmFitOfflineMasterModel(self.master_id, args)

    def fit_online(self, batch_filenames=None, batch_weights=None, update_after=None,
                   apply_weight=None, decay_weight=None, asynchronous=None):
        """
        :param batch_filenames: name of batches to process
        :type batch_filenames: list of str
        :param batch_weights: weights of batches to process
        :type batch_weights: list of float
        :param update_after: number of batches to be passed for Phi synchronizations
        :type update_after: list of int
        :param apply_weight: weight of applying new counters\
                (len == len of update_after)
        :type apply_weight: list of float
        :param decay_weight: weight of applying old counters\
                (len == len of update_after)
        :type decay_weight: list of float
        :param bool asynchronous: whether to use the asynchronous implementation\
                of the EM-algorithm or not
        """
        args = messages.FitOnlineMasterModelArgs()
        if batch_filenames is not None:
            args.ClearField('batch_filename')
            for filename in batch_filenames:
                args.batch_filename.append(filename)

        if batch_weights is not None:
            args.ClearField('batch_weight')
            for weight in batch_weights:
                args.batch_weight.append(weight)

        if update_after is not None:
            args.ClearField('update_after')
            for value in update_after:
                args.update_after.append(value)

        if update_after is not None:
            args.ClearField('update_after')
            for value in update_after:
                args.update_after.append(value)

        if apply_weight is not None:
            args.ClearField('apply_weight')
            for value in apply_weight:
                args.apply_weight.append(value)

        if decay_weight is not None:
            args.ClearField('decay_weight')
            for value in decay_weight:
                args.decay_weight.append(value)

        if asynchronous is not None:
            args.asynchronous = asynchronous

        self._lib.ArtmFitOnlineMasterModel(self.master_id, args)

    def transform(self, batches=None, batch_filenames=None, theta_matrix_type=None,
                  predict_class_id=None):
        """
        :param batches: list of Batch instances
        :param batch_weights: weights of batches to transform
        :type batch_weights: list of float
        :param int theta_matrix_type: type of matrix to be returned
        :param predict_class_id: class_id of a target modality to predict
        :type predict_class_id: str, default None
        :return: messages.ThetaMatrix object
        """
        args = messages.TransformMasterModelArgs()
        if batches is not None:
            args.ClearField('batch')
            for batch in batches:
                batch_ref = args.batch.add()
                batch_ref.CopyFrom(batch)

        if batch_filenames is not None:
            args.ClearField('batch_filename')
            for filename in batch_filenames:
                args.batch_filename.append(filename)

        if theta_matrix_type is not None:
            args.theta_matrix_type = theta_matrix_type

        if predict_class_id is not None:
            args.predict_class_id = predict_class_id

        if theta_matrix_type not in [constants.ThetaMatrixType_None, constants.ThetaMatrixType_Cache]:
            theta_matrix_info = self._lib.ArtmRequestTransformMasterModelExternal(self.master_id, args)

            num_rows = len(theta_matrix_info.item_id)
            num_cols = theta_matrix_info.num_topics
            numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)
            self._lib.ArtmCopyRequestedObject(numpy_ndarray)

            return theta_matrix_info, numpy_ndarray
        else:
            self._lib.ArtmRequestTransformMasterModel(self.master_id, args)
            return None, None

    def import_batches(self, batches=None):
        """
        :param list batches: list of BigARTM batches loaded into RAM
        """
        args = messages.ImportBatchesArgs()
        if batches is not None:
            args.ClearField('batch')
            for batch in batches:
                batch_ref = args.batch.add()
                batch_ref.CopyFrom(batch)
        self._lib.ArtmImportBatches(self.master_id, args)

    def remove_batch(self, batch_id=None):
        """
        :param unicode batch_id: id of batch, loaded in RAM
        """
        if batch_id is not None:
            self._lib.ArtmDisposeBatch(self.master_id, batch_id)

    def export_score_tracker(self, filename):
        """
        :param str filename: the name of file to save score tracker into binary format
        """
        args = messages.ExportScoreTrackerArgs(file_name=filename)
        result = self._lib.ArtmExportScoreTracker(self.master_id, args)

    def import_score_tracker(self, filename):
        """
        :param str filename: the name of file to load score tracker from binary format
        """
        args = messages.ImportScoreTrackerArgs(file_name=filename)
        result = self._lib.ArtmImportScoreTracker(self.master_id, args)
