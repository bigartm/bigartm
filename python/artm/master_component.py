import os
import numpy
import codecs

from .wrapper import messages_pb2 as messages
from .wrapper import constants


def _regularizer_type(config):
    if isinstance(config, messages.SmoothSparseThetaConfig):
        return constants.RegularizerConfig_Type_SmoothSparseTheta
    elif isinstance(config, messages.SmoothSparsePhiConfig):
        return constants.RegularizerConfig_Type_SmoothSparsePhi
    elif isinstance(config, messages.DecorrelatorPhiConfig):
        return constants.RegularizerConfig_Type_DecorrelatorPhi
    elif isinstance(config, messages.LabelRegularizationPhiConfig):
        return constants.RegularizerConfig_Type_LabelRegularizationPhi
    elif isinstance(config, messages.SpecifiedSparsePhiConfig):
        return constants.RegularizerConfig_Type_SpecifiedSparsePhi
    elif isinstance(config, messages.ImproveCoherencePhiConfig):
        return constants.RegularizerConfig_Type_ImproveCoherencePhi
    elif isinstance(config, messages.SmoothPtdwConfig):
        return constants.RegularizerConfig_Type_SmoothPtdw
    elif isinstance(config, messages.TopicSelectionThetaConfig):
        return constants.RegularizerConfig_Type_TopicSelectionTheta


def _score_type(config):
    if isinstance(config, messages.PerplexityScoreConfig):
        return constants.ScoreConfig_Type_Perplexity
    elif isinstance(config, messages.SparsityThetaScoreConfig):
        return constants.ScoreConfig_Type_SparsityTheta
    elif isinstance(config, messages.SparsityPhiScoreConfig):
        return constants.ScoreConfig_Type_SparsityPhi
    elif isinstance(config, messages.ItemsProcessedScoreConfig):
        return constants.ScoreConfig_Type_ItemsProcessed
    elif isinstance(config, messages.TopTokensScoreConfig):
        return constants.ScoreConfig_Type_TopTokens
    elif isinstance(config, messages.ThetaSnippetScoreConfig):
        return constants.ScoreConfig_Type_ThetaSnippet
    elif isinstance(config, messages.TopicKernelScoreConfig):
        return constants.ScoreConfig_Type_TopicKernel
    elif isinstance(config, messages.TopicMassPhiScoreConfig):
        return constants.ScoreConfig_Type_TopicMassPhi
    elif isinstance(config, messages.ClassPrecisionScoreConfig):
        return constants.ScoreConfig_Type_ClassPrecision


def _score_data_func(score_data_type):
        if score_data_type == constants.ScoreData_Type_Perplexity:
            return messages.PerplexityScore
        elif score_data_type == constants.ScoreData_Type_SparsityTheta:
            return messages.SparsityThetaScore
        elif score_data_type == constants.ScoreData_Type_SparsityPhi:
            return messages.SparsityPhiScore
        elif score_data_type == constants.ScoreData_Type_ItemsProcessed:
            return messages.ItemsProcessedScore
        elif score_data_type == constants.ScoreData_Type_TopTokens:
            return messages.TopTokensScore
        elif score_data_type == constants.ScoreData_Type_ThetaSnippet:
            return messages.ThetaSnippetScore
        elif score_data_type == constants.ScoreData_Type_TopicKernel:
            return messages.TopicKernelScore
        elif score_data_type == constants.ScoreData_Type_TopicMassPhi:
            return messages.TopicMassPhiScore
        elif score_data_type == constants.ScoreData_Type_ClassPrecision:
            return messages.ClassPrecisionScore


def _prepare_config(topic_names, class_ids, scores, regularizers, num_processors,
                    pwt_name, nwt_name, num_document_passes, reuse_theta, cache_theta, args=None):
        master_config = messages.MasterModelConfig()

        if args is not None:
            master_config.CopyFrom(args)

        if topic_names is not None:
            master_config.ClearField('topic_name')
            for topic_name in topic_names:
                master_config.topic_name.append(topic_name)

        if class_ids is not None:
            master_config.ClearField('class_id')
            master_config.ClearField('class_weight')
            for class_id, class_weight in class_ids.iteritems():
                master_config.class_id.append(class_id)
                master_config.class_weight.append(class_weight)

        if scores is not None:
            master_config.ClearField('score_config')
            for name, config in scores.iteritems():
                score_config = master_config.score_config.add()
                score_config.name = name
                score_config.type = _score_type(config)
                score_config.config = config.SerializeToString()

        if regularizers is not None:
            master_config.ClearField('regularizer_config')
            for name, config_tau in regularizers.iteritems():
                regularizer_config = master_config.regularizer_config.add()
                regularizer_config.name = name
                regularizer_config.type = _regularizer_type(config)
                regularizer_config.config = config_tau[0].SerializeToString()
                regularizer_config.tau = config_tau[1]

        if num_processors is not None:
            master_config.threads = num_processors

        if reuse_theta is not None:
            master_config.reuse_theta = reuse_theta

        if cache_theta is not None:
            master_config.cache_theta = cache_theta

        if pwt_name is not None:
            master_config.pwt_name = pwt_name

        if nwt_name is not None:
            master_config.nwt_name = nwt_name

        if num_document_passes is not None:
            master_config.inner_iterations_count = num_document_passes

        return master_config


class MasterComponent(object):
    def __init__(self, library, topic_names=None, class_ids=None, scores=None, regularizers=None,
                 num_processors=None, pwt_name=None, nwt_name=None, num_document_passes=None,
                 reuse_theta=None, cache_theta=False):
        """ Args:
            - library: an instance of LibArtm
            - topic_names (list of str): list of topic names to use in model
            - class_ids (dict): key - class_id, value - class_weight
            - scores (dict): key - score name, value - config
            - regularizers (dict): key - regularizer name, value - pair (config, tau)
            - num_processors (int): number of work threads to use
            - pwt_name (str): name of pwt matrix
            - nwt_name (str): name of nwt matrix
            - num_document_passes (int): num passes through each document
            - reuse_theta (bool): reuse Theta from previous iteration or not
            - cache_theta (bool): save or not the Theta matrix
        """
        self._lib = library

        master_config = _prepare_config(topic_names=topic_names,
                                        class_ids=class_ids,
                                        scores=scores,
                                        regularizers=regularizers,
                                        num_processors=num_processors,
                                        pwt_name=pwt_name,
                                        nwt_name=nwt_name,
                                        num_document_passes=num_document_passes,
                                        reuse_theta=reuse_theta,
                                        cache_theta=cache_theta)

        self._config = master_config
        self.master_id = self._lib.ArtmCreateMasterModel(master_config)

    def reconfigure(self, topic_names=None, class_ids=None, scores=None, regularizers=None,
                    num_processors=None, pwt_name=None, nwt_name=None, num_document_passes=None,
                    reuse_theta=None, cache_theta=False):
        master_config = _prepare_config(topic_names=topic_names,
                                        class_ids=class_ids,
                                        scores=scores,
                                        regularizers=regularizers,
                                        num_processors=num_processors,
                                        pwt_name=pwt_name,
                                        nwt_name=nwt_name,
                                        num_document_passes=num_document_passes,
                                        reuse_theta=reuse_theta,
                                        cache_theta=cache_theta,
                                        args=self._config)

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def import_dictionary(self, filename, dictionary_name):
        """Args:
           - filename(str): full name of dictionary file
           - dictionary_name(str): name of imported dictionary
        """
        args = messages.ImportDictionaryArgs(dictionary_name=dictionary_name, file_name=filename)
        self._lib.ArtmImportDictionary(self.master_id, args)

    def export_dictionary(self, filename, dictionary_name):
        """Args:
           - filename(str): full name for dictionary file
           - dictionary_name(str): name of exported dictionary
        """
        args = messages.ExportDictionaryArgs(dictionary_name=dictionary_name, file_name=filename)
        self._lib.ArtmExportDictionary(self.master_id, args)

    def create_dictionary(self, dictionary_data, dictionary_name=None):
        """Args:
           - dictionary_data: an instance of DictionaryData with info
             about dictionary
           - dictionary_name(str): name of exported dictionary
        """

        if dictionary_name is not None:
            dictionary_data.name = dictionary_name

        self._lib.ArtmCreateDictionary(self.master_id, dictionary_data)

    def get_dictionary(self, dictionary_name):
        """Args:
           - dictionary_name(str): name of dictionary to get
        """
        args = messages.GetDictionaryArgs(dictionary_name=dictionary_name)
        dictionary_data = self._lib.ArtmRequestDictionary(self.master_id, args)
        return dictionary_data

    def gather_dictionary(self, dictionary_target_name=None, data_path=None, cooc_file_path=None,
                          vocab_file_path=None, symmetric_cooc_values=None, args=None):
        """Args:
           - dictionary_target_name(str): name of the dictionary in the core
           - data_path(str): full path to batches folder
           - cooc_file_path(str): full path to the file with cooc info
           - vocab_file_path(str): full path to the file with vocabulary
           - symmetric_cooc_values(str): if the cooc matrix should
             considered to be symmetric or not
           - args: an instance of GatherDictionaryArgs
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
                          args=None):

        """Args:
           - dictionary_name(str): name of the dictionary in the core to filter
           - dictionary_target_name(str): name for the new filtered dictionary in the core
           - class_id(str): class_id to filter
           - min_df(float): min df value to pass the filter
           - max_df(float): max df value to pass the filter
           - min_df_rate (float): min df rate to pass the filter
           - max_df_rate (float): max df rate to pass the filter
           - min_tf(float): min tf value to pass the filter
           - max_tf(float): max tf value to pass the filter
           - args: an instance of FilterDictionaryArgs
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

        self._lib.ArtmFilterDictionary(self.master_id, filter_args)

    def initialize_model(self, model_name=None, num_topics=None, topic_names=None,
                         dictionary_name=None, seed=None, args=None):
        """Args:
           - model_name(str): name of pwt matrix in BigARTM
           - num_topics(int): number of topics in model
           - topic_names(list of str): the list of names of topics to be used in model
           - dictionary_name(str): name of imported dictionary
           - seed (unsigned int or -1): seed for random initialization, default=None (no seed)
           - args: an instance of InitilaizeModelArgs
        """
        init_args = messages.InitializeModelArgs()
        if args is not None:
            init_args = args
        if model_name is not None:
            init_args.model_name = model_name
        if seed is not None:
            init_args.seed = seed
        if num_topics is not None:
            init_args.topics_count = num_topics
        if topic_names is not None:
            init_args.ClearField('topic_name')
            for topic_name in topic_names:
                init_args.topic_name.append(topic_name)

        init_args.dictionary_name = dictionary_name
        self._lib.ArtmInitializeModel(self.master_id, init_args)

    def process_batches(self, pwt, nwt, num_inner_iterations=None, batches_folder=None,
                        batches=None, regularizer_name=None, regularizer_tau=None,
                        class_ids=None, class_weights=None, find_theta=False,
                        reset_scores=False, reuse_theta=False, find_ptdw=False,
                        predict_class_id=None):
        """Args:
           - pwt(str): name of pwt matrix in BigARTM
           - nwt(str): name of nwt matrix in BigARTM
           - num_inner_iterations(int): number of inner iterations during processing
           - batches_folder(str): full path to data folder (alternative 1)
           - batches(list of str): full file names of batches to process (alternative 2)
           - regularizer_name(list of str): list of names of Theta regularizers to use
           - regularizer_tau(list of double): list of tau coefficients for Theta regularizers
           - class_ids(list of str): list of class ids to use during processing
           - class_weights(list of double): list of corresponding weights of class ids
           - find_theta(bool): find theta matrix for 'batches' (if alternative 2)
           - reset_scores(bool): reset scores after iterations or not
           - reuse_theta(bool): initialize by theta from previous collection pass
           - find_ptdw(bool): count and return Ptdw matrix or not (works if find_theta == False)
           - predict_class_id(str): class_id of a target modality to predict (default = None)
           Returns:
           - tuple (messages.ThetaMatrix, numpy.ndarray) --- the info about Theta (find_theta==True)
           - messages.ThetaMatrix --- the info about Theta (find_theta==False)
        """
        args = messages.ProcessBatchesArgs(pwt_source_name=pwt,
                                           reset_scores=reset_scores,
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

        if num_inner_iterations is not None:
            args.inner_iterations_count = num_inner_iterations

        if regularizer_name is not None and regularizer_tau is not None:
            for name, tau in zip(regularizer_name, regularizer_tau):
                args.regularizer_name.append(name)
                args.regularizer_tau.append(tau)

        if class_ids is not None and class_weights is not None:
            for class_id, weight in zip(class_ids, class_weights):
                args.class_id.append(class_id)
                args.class_weight.append(weight)

        if predict_class_id is not None:
            args.predict_class_id = predict_class_id

        func = None
        if find_theta or find_ptdw:
            args.theta_matrix_type = constants.ProcessBatchesArgs_ThetaMatrixType_Dense
            if find_ptdw:
                args.theta_matrix_type = constants.ProcessBatchesArgs_ThetaMatrixType_DensePtdw
            func = self._lib.ArtmRequestProcessBatchesExternal
        elif not find_theta or find_theta is None:
            func = self._lib.ArtmRequestProcessBatches

        result = func(self.master_id, args)

        if not find_theta and not find_ptdw:
            return result.theta_matrix

        num_rows = len(result.theta_matrix.item_id)
        num_cols = result.theta_matrix.topics_count
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)

        cp_args = messages.CopyRequestResultArgs()
        cp_args.request_type = constants.CopyRequestResultArgs_RequestType_GetThetaSecondPass
        self._lib.ArtmCopyRequestResultEx(numpy_ndarray, cp_args)

        return result.theta_matrix, numpy_ndarray

    def regularize_model(self, pwt, nwt, rwt, regularizer_name, regularizer_tau):
        """Args:
           - pwt(str): name of pwt matrix in BigARTM
           - nwt(str): name of nwt matrix in BigARTM
           - rwt(str): name of rwt matrix in BigARTM
           - regularizer_name(list of str): list of names of Phi regularizers to use
           - regularizer_tau(list of double): list of tau coefficients for Phi regularizers
        """
        args = messages.RegularizeModelArgs(pwt_source_name=pwt,
                                            nwt_source_name=nwt,
                                            rwt_target_name=rwt)

        for name, tau in zip(regularizer_name, regularizer_tau):
            reg_set = args.regularizer_settings.add()
            reg_set.name = name
            reg_set.tau = tau
            reg_set.use_relative_regularization = False

        self._lib.ArtmRegularizeModel(self.master_id, args)

    def normalize_model(self, pwt, nwt, rwt=None):
        """Args:
           - pwt(str): name of pwt matrix in BigARTM
           - nwt(str): name of nwt matrix in BigARTM
           - rwt(str): name of rwt matrix in BigARTM
        """
        args = messages.NormalizeModelArgs(pwt_target_name=pwt, nwt_source_name=nwt)
        if rwt is not None:
            args.rwt_source_name = rwt

        self._lib.ArtmNormalizeModel(self.master_id, args)

    def merge_model(self, models, nwt, topic_names=None):
        """ MasterComponent.MergeModel() --- merge multiple nwt-increments together.
        Args:
        - models(dict): list of models with nwt-increments and their weights,
                        key - nwt_source_name, value - source_weight.
        - nwt(str): the name of target matrix to store combined nwt.
                    The matrix will be created by this operation.
        - topic_names(list of str): names of topics in the resulting model. By default model
                                    names are taken from the first model in the list.
        """
        args = messages.MergeModelArgs(nwt_target_name=nwt)
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)
        for nwt_source_name, source_weight in models.iteritems():
            args.nwt_source_name.append(nwt_source_name)
            args.source_weight.append(source_weight)

        self._lib.ArtmMergeModel(self.master_id, args)

    def attach_model(self, model):
        """Args:
           - model(str): name of matrix in BigARTM
           Returns:
           - messages.TopicModel() object with info about Phi matrix
           - numpy.ndarray with Phi data (e.g. p(w|t) values)
        """
        topics = self.get_phi_info(model, constants.GetTopicModelArgs_RequestType_TopicNames)
        tokens = self.get_phi_info(model, constants.GetTopicModelArgs_RequestType_Tokens)

        num_rows = len(tokens.token)
        num_cols = topics.topics_count
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)

        self._lib.ArtmAttachModel(self.master_id,
                                  messages.AttachModelArgs(model_name=model),
                                  numpy_ndarray)

        topic_model = messages.TopicModel(topics_count=topics.topics_count)
        topic_model.topic_name.MergeFrom(topics.topic_name)
        topic_model.class_id.MergeFrom(tokens.class_id)
        topic_model.token.MergeFrom(tokens.token)

        return topic_model, numpy_ndarray

    def create_regularizer(self, name, config, tau):
        """Args:
           - name(str): the name of the future regularizer
           - config: the config of the future regularizer
           - tau(float): the coefficient of the regularization
        """
        master_config = messages.MasterModelConfig()
        master_config.CopyFrom(self._config)

        regularizer_config = master_config.regularizer_config.add()
        regularizer_config.name = name
        regularizer_config.type = _regularizer_type(config)
        regularizer_config.config = config.SerializeToString()
        regularizer_config.tau = tau

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def reconfigure_regularizer(self, name, config=None, tau=None):
        master_config = messages.MasterModelConfig()
        master_config.CopyFrom(self._config)

        for index, regularizer_config in enumerate(master_config.regularizer_config):
            if regularizer_config.name == name:
                if config is not None:
                    master_config.regularizer_config[index].config = config.SerializeToString()
                if tau is not None:
                    master_config.regularizer_config[index].tau = tau

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def create_score(self, name, config):
        """Args:
           - name(str): the name of the future score
           - config: an instance of ***ScoreConfig
        """
        master_config = messages.MasterModelConfig()
        master_config.CopyFrom(self._config)

        score_config = master_config.score_config.add()
        score_config.name = name
        score_config.type = _score_type(config)
        score_config.config = config.SerializeToString()

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def get_score(self, model_name, score_name):
        """Args:
           - model_name(str): name of pwt matrix in BigARTM
           - score_name(str): the user defined name of score to retrieve
           - score_config: reference to score data object
        """
        args = messages.GetScoreValueArgs(model_name=model_name, score_name=score_name)
        score_data = self._lib.ArtmRequestScore(self.master_id, args)

        score_info = _score_data_func(score_data.type)()
        score_info.ParseFromString(score_data.data)

        return score_info

    def reconfigure_score(self, name, config):
        master_config = messages.MasterModelConfig()
        master_config.CopyFrom(self._config)

        for index, score_config in enumerate(master_config.score_config):
            if score_config.name == name:
                master_config.score_config[index].config = config.SerializeToString()

        self._config = master_config
        self._lib.ArtmReconfigureMasterModel(self.master_id, master_config)

    def get_theta_info(self, model):
        """Args:
           - model(str): name of matrix in BigARTM
           Returns:
           - messages.ThetaMatrix object
        """
        args = messages.GetThetaMatrixArgs(model_name=model)
        args.eps = 1.001  # hack to not get any data back
        args.matrix_layout = 1  # GetThetaMatrixArgs_MatrixLayout_Sparse
        theta_matrix_info = self._lib.ArtmRequestThetaMatrix(self.master_id, args)

        return theta_matrix_info

    def get_theta_matrix(self, model, clean_cache=None, topic_names=None):
        """Args:
           - model(str): name of matrix in BigARTM
           - cleab_cache(bool): remove or not the info about Theta after retrieval
           - topic_names(list of str): list of topics to retrieve (None == all topics)
           Returns:
           - numpy.ndarray with Theta data (e.g. p(t|d) values)
        """
        args = messages.GetThetaMatrixArgs(model_name=model)
        if clean_cache is not None:
            args.clean_cache = clean_cache
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)

        theta_matrix_info = self._lib.ArtmRequestThetaMatrixExternal(self.master_id, args)

        num_rows = len(theta_matrix_info.item_id)
        num_cols = theta_matrix_info.topics_count
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)

        cp_args = messages.CopyRequestResultArgs()
        cp_args.request_type = constants.CopyRequestResultArgs_RequestType_GetThetaSecondPass
        self._lib.ArtmCopyRequestResultEx(numpy_ndarray, cp_args)

        return theta_matrix_info, numpy_ndarray

    def get_phi_info(self, model, request_type=None):
        """Args:
           - model(str): name of matrix in BigARTM
           - request_type(int): Pwt = 0 | Nwt = 1; | TopicNames = 2 | Tokens = 3
           Returns:
           - messages.TopicModel object
        """
        args = messages.GetTopicModelArgs(model_name=model)
        if request_type is not None:
            args.request_type = request_type

        phi_matrix_info = self._lib.ArtmRequestTopicModel(self.master_id, args)

        return phi_matrix_info

    def get_phi_matrix(self, model, topic_names=None, class_ids=None, use_sparse_format=None):
        """Args:
           - model(str): name of matrix in BigARTM
           - topic_names(list of str): list of topics to retrieve (None == all topics)
           - class_ids(list of str): list of class ids to retrieve (None == all class ids)
           - use_sparse_format(bool): use sparse\dense layout
           Returns:
           - numpy.ndarray with Phi data (e.g. p(w|t) values)
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
            args.matrix_layout = constants.GetTopicModelArgs_MatrixLayout_Sparse

        phi_matrix_info = self._lib.ArtmRequestTopicModelExternal(self.master_id, args)

        num_rows = len(phi_matrix_info.token)
        num_cols = phi_matrix_info.topics_count
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)

        cp_args = messages.CopyRequestResultArgs()
        cp_args.request_type = constants.CopyRequestResultArgs_RequestType_GetModelSecondPass
        self._lib.ArtmCopyRequestResultEx(numpy_ndarray, cp_args)

        return phi_matrix_info, numpy_ndarray

    def export_model(self, model, filename):
        args = messages.ExportModelArgs(model_name=model, file_name=filename)
        result = self._lib.ArtmExportModel(self.master_id, args)

    def import_model(self, model, filename):
        """Args:
           - model(str): name of matrix in BigARTM
           - filename(str): the name of file to load model from binary format
        """
        args = messages.ImportModelArgs(model_name=model, file_name=filename)
        result = self._lib.ArtmImportModel(self.master_id, args)

    def get_info(self):
        info = self._lib.ArtmRequestMasterComponentInfo(self.master_id,
                                                        messages.GetMasterComponentInfoArgs())
        return info

    def fit_offline(self, batch_filenames=None, batch_weights=None,
                    num_collection_passes=None, batches_folder=None):
        """Args:
           - batch_filenames(list of str): name of batches to process
           - batch_weights(list of float): weights of batches to process
           - num_collection_passes(int): number of outer iterations
           - batches_folder(str): folder containing batches to process
        """
        args = messages.FitOfflineMasterModelArgs()
        if batch_filenames is not None:
            args.ClearField('batch_filename')
            for filename in batch_filenames:
                args.batch_filename.append(filename)

        if batch_weights is not None:
            args.ClearField('batch_weight')
            for weight in batch_weights:
                args.batch_weight.append(weight)

        if num_collection_passes is not None:
            args.passes = num_collection_passes

        if batches_folder is not None:
            args.batch_folder = batches_folder

        self._lib.ArtmFitOfflineMasterModel(self.master_id, args)

    def fit_online(self, batch_filenames=None, batch_weights=None, update_after=None,
                   apply_weight=None, decay_weight=None, async=None):
        """Args:
           - batch_filenames(list of str): name of batches to process
           - batch_weights(list of float): weights of batches to process
           - update_after(list of int): number of batches to be passed for Phi synchronizations
           - apply_weight(list of float): weight of applying new counters
             (len == len of update_after)
           - decay_weight(list of float): weight of applying old counters
             (len == len of update_after)
           - async(bool): use or not the async implementation of the EM-algorithm
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

        if async is not None:
            args.async = async

        self._lib.ArtmFitOnlineMasterModel(self.master_id, args)

    def transform(self, batches=None, batch_filenames=None,
                  theta_matrix_type=None, predict_class_id=None):
        """Args:
           - batches(list of batches): list of instances of Batch
           - batch_weights(list of float): weights of batches to transform
           - theta_matrix_type(int): type of matrix to be returned
           - predict_class_id(int): type of matrix to be returned
           Returns:
           - messages.ThetaMatrix object
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

        theta_matrix_info = self._lib.ArtmRequestTransformMasterModelExternal(self.master_id, args)

        num_rows = len(theta_matrix_info.item_id)
        num_cols = theta_matrix_info.topics_count
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)

        cp_args = messages.CopyRequestResultArgs()
        cp_args.request_type = constants.CopyRequestResultArgs_RequestType_GetThetaSecondPass
        self._lib.ArtmCopyRequestResultEx(numpy_ndarray, cp_args)

        return theta_matrix_info, numpy_ndarray
