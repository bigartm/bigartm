import os
import numpy

import wrapper
import wrapper.messages_pb2 as messages
import wrapper.constants as constants


class MasterComponent(object):
    def __init__(self, library, num_processors=None, scores=None, cache_theta=False):
        """ Args:
            - library: an instance of LibArtm
            - num_processors (int): number of work threads to use
            - scores (list): list of tuples (name, config) for each config to use
            - cache_theta (bool): save or not the Theta matrix
        """
        self._lib = library
        master_config = messages.MasterComponentConfig()

        if num_processors is not None:
            master_config.processors_count = num_processors
        if cache_theta is not None:
            master_config.cache_theta = cache_theta
        if scores is not None:
            for name, config in scores:
                ref_score_config = master_config.score_config.add()
                ref_score_config.name = name
            
                if isinstance(config, messages.PerplexityScoreConfig):
                    ref_score_config.type = constants.ScoreConfig_Type_Perplexity
                elif isinstance(config, messages.SparsityThetaScoreConfig):
                    ref_score_config.type = constants.ScoreConfig_Type_SparsityTheta
                elif isinstance(config, messages.SparsityPhiScoreConfig):
                    ref_score_config.type = constants.ScoreConfig_Type_SparsityPhi
                elif isinstance(config, messages.ItemsProcessedScoreConfig):
                    ref_score_config.type = constants.ScoreConfig_Type_ItemsProcessed
                elif isinstance(config, messages.TopTokensScoreConfig):
                    ref_score_config.type = constants.ScoreConfig_Type_TopTokens
                elif isinstance(config, messages.ThetaSnippetScoreConfig):
                    ref_score_config.type = constants.ScoreConfig_Type_ThetaSnippet
                elif isinstance(config, messages.TopicKernelScoreConfig):
                    ref_score_config.type = constants.ScoreConfig_Type_TopicKernel

                ref_score_config.config = config.SerializeToString()

        self.master_id = self._lib.ArtmCreateMasterComponent(master_config)

    def import_dictionary(self, file_name, dictionary_name):
        """Args:
           - file_name(str): full name of dictionary file
           - dictionary_name(str): name of imported dictionary
        """
        args = messages.ImportDictionaryArgs()
        args.dictionary_name = dictionary_name
        args.file_name = file_name

        self._lib.ArtmImportDictionary(self.master_id, args)

    def initialize_model(self, model_name=None, num_topics=None, source_type=None, disk_path=None,
                         dictionary_name=None, args=None):
        """Args:
           - model_name(str): name of pwt matrix in BigARTM
           - source_type(str): 'batches' | 'dictionary'
           - num_topics(int): number of topics in model
           - disk_path(str): full name of folder with batches (need if InitializeModelArgs_SourceType_Batches)
           - dictionary_name(str): name of imported dictionary (need if InitializeModelArgs_SourceType_Dictionary)
        """
        init_args = messages.InitializeModelArgs()
        if args is not None:
            init_args = args
        if model_name is not None:
            init_args.model_name = model_name
        if num_topics is not None:
            init_args.topics_count = num_topics

        if source_type is 'batches':
            init_args.source_type = constants.InitializeModelArgs_SourceType_Batches
            if disk_path is not None:
                init_args.disk_path = disk_path
        elif source_type is 'dictionary':
            init_args.source_type = constants.InitializeModelArgs_SourceType_Dictionary
            if dictionary_name is not None:
                init_args.dictionary_name = dictionary_name

        self._lib.ArtmInitializeModel(self.master_id, init_args)

    def retrieve_score(self, model_name, score_name):
        """Args:
           - model_name(str): name of pwt matrix in BigARTM
           - score_name(str): the user defined name of score to retrieve
           - score_config: reference to score data object
        """
        args = messages.GetScoreValueArgs()
        args.model_name = model_name
        args.score_name = score_name

        results = self._lib.ArtmRequestScore(self.master_id, args)
        score_data = messages.ScoreData()
        score_data.ParseFromString(results)

        score_info = None
        if score_data.type == constants.ScoreData_Type_Perplexity:
            score_info = messages.PerplexityScore()
        elif score_data.type == constants.ScoreData_Type_SparsityTheta:
            score_info = messages.SparsityThetaScore()
        elif score_data.type == constants.ScoreData_Type_SparsityPhi:
            score_info = messages.SparsityPhiScore()
        elif score_data.type == constants.ScoreData_Type_ItemsProcessed:
            score_info = messages.ItemsProcessedScore()
        elif score_data.type == constants.ScoreData_Type_TopTokens:
            score_info = messages.TopTokensScore()
        elif score_data.type == constants.ScoreData_Type_ThetaSnippet:
            score_info = messages.ThetaSnippetScore()
        elif score_data.type == constants.ScoreData_Type_TopicKernel:
            score_info = messages.TopicKernelScore()
        score_info.ParseFromString(score_data.data)
        return score_info

    def process_batches(self, pwt, nwt, num_inner_iterations, batches_folder=None,
                        batches=None, regularizer_name=None, regularizer_tau=None,
                        class_ids=None, class_weights=None, find_theta=False):
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
           - find_theat(bool): find theta matrix for 'batches' (if alternative 2)
           Returns:
           - tuple (messages.ThetaMatrix, numpy.ndarray) --- the info about Theta (find_theta==True)
           - messages.ThetaMatrix --- the info about Theta (find_theta==False)
        """
        args = messages.ProcessBatchesArgs()
        args.pwt_source_name = pwt
        args.nwt_target_name = nwt
        if batches_folder is not None:
            for name in os.listdir(batches_folder):
                _, extension = os.path.splitext(name)
                if extension == '.batch':
                    args.batch_filename.append(os.path.join(batches_folder, name))
        if batches is not None:
            for batch in batches:
                args.batch_filename.append(batch)
            
        args.inner_iterations_count = num_inner_iterations

        if regularizer_name is not None and regularizer_tau is not None:
            for name, tau in zip(regularizer_name, regularizer_tau):
                args.regularizer_name.append(name)
                args.regularizer_tau.append(tau)

        if class_ids is not None and class_weights is not None:
            for class_id, weight in zip(class_ids, class_weights):
                args.class_id.append(class_id)
                args.class_weight.append(weight)

        func = None
        if find_theta:
            args.theta_matrix_type = constants.ProcessBatchesArgs_ThetaMatrixType_Dense
            func = self._lib.ArtmRequestProcessBatchesExternal
        elif not find_theta or find_theta is None:
            func = self._lib.ArtmRequestProcessBatches

        retval = func(self.master_id, args)

        result = messages.ProcessBatchesResult()
        result.ParseFromString(retval)

        if not find_theta:
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
        args = messages.RegularizeModelArgs()
        args.pwt_source_name = pwt
        args.nwt_source_name = nwt
        args.rwt_target_name = rwt
        
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
        args = messages.NormalizeModelArgs()
        args.pwt_target_name = pwt
        args.nwt_source_name = nwt
        if rwt is not None:
            args.rwt_source_name = rwt

        self._lib.ArtmNormalizeModel(self.master_id, args)

    def merge_model(self, models, nwt, topic_names=None):
        """ MasterComponent.MergeModel() --- merge multiple nwt-increments together.
        Args:
        - models(dict): list of models with nwt-increments and their weights,
                        key - nwt_source_name, value - source_weight.
        - nwt(str): the name of target matrix to store combined nwt. The matrix will be created by this operation.
        - topic_names(list of str): names of topics in the resulting model. By default model
                                    names are taken from the first model in the list.
        """
        args = messages.MergeModelArgs()
        args.nwt_target_name = nwt
        if topic_names is not None:
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

        self._lib.ArtmAttachModel(self.master_id, messages.AttachModelArgs(model_name=model), numpy_ndarray)

        topic_model = messages.TopicModel()
        topic_model.topics_count = topics.topics_count
        topic_model.topic_name.MergeFrom(topics.topic_name)
        topic_model.class_id.MergeFrom(tokens.class_id)
        topic_model.token.MergeFrom(tokens.token)

        return topic_model, numpy_ndarray

    def create_regularizer(self, name, type, config):
        """Args:
           - name(str): the name of the future regularizer
           - type(int): the type of the future regularizer
           - config: an instance of ***RegularizerConfig
        """
        cfg = messages.RegularizerConfig(name=name, type=type, config=config.SerializeToString())
        self._lib.ArtmCreateRegularizer(self.master_id, cfg)

    def create_smooth_sparse_phi_regularizer(self, name, class_ids=None, dictionary_name=None):
        """Args:
           - name(str): the name of the future regularizer
           - class_ids(list of str): the list of class_ids to be regularized
           - dictionary_name(str): name of imported dictionary
        """
        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = name
        ref_reg_config.type = constants.RegularizerConfig_Type_SmoothSparsePhi
        
        config = messages.SmoothSparsePhiConfig()
        if class_ids is not None:
            for class_id in class_ids:
                config.class_id.append(class_id)
        if dictionary_name is not None:
            config.dictionary_name = dictionary_name

        ref_reg_config.config = config.SerializeToString()
        self._lib.ArtmCreateRegularizer(self.master_id, ref_reg_config)

    def create_smooth_sparse_theta_regularizer(self, name):
        """Args:
           - name(str): the name of the future regularizer
        """
        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = name
        ref_reg_config.type = constants.RegularizerConfig_Type_SmoothSparseTheta
        ref_reg_config.config = messages.SmoothSparseThetaConfig().SerializeToString()

        self._lib.ArtmCreateRegularizer(self.master_id, ref_reg_config)

    def create_decorrelator_phi_regularizer(self, name, class_ids=None):
        """Args:
           - name(str): the name of the future regularizer
           - class_ids(list of str): the list of class_ids to be regularized
        """
        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = name
        ref_reg_config.type = constants.RegularizerConfig_Type_DecorrelatorPhi
        
        config = messages.DecorrelatorPhiConfig()
        if class_ids is not None:
            for class_id in class_ids:
                config.class_id.append(class_id)
        
        ref_reg_config.config = config.SerializeToString()
        self._lib.ArtmCreateRegularizer(self.master_id, ref_reg_config)

    def get_theta_info(self, model):
        """Args:
           - model(str): name of matrix in BigARTM
           Returns:
           - messages.ThetaMatrix object
        """
        result = self._lib.ArtmRequestThetaMatrix(self.master_id, messages.GetThetaMatrixArgs(model_name=model))

        theta_matrix_info = messages.ThetaMatrix()
        theta_matrix_info.ParseFromString(result)

        return theta_matrix_info

    def get_theta_matrix(self, model, clean_cache=None, topic_names=None):
        """Args:
           - model(str): name of matrix in BigARTM
           - cleab_cache(bool): remove or not the info about Theta after retrieval
           - topic_names(list of str): list of topics to retrieve (None == all topics)
           Returns:
           - numpy.ndarray with Theta data (e.g. p(t|d) values)
        """
        args = messages.GetThetaMatrixArgs()
        args.model_name = model
        if clean_cache is not None:
            args.clean_cache = clean_cache
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)

        result = self._lib.ArtmRequestThetaMatrixExternal(self.master_id, args)

        theta_matrix_info = messages.ThetaMatrix()
        theta_matrix_info.ParseFromString(result)

        num_rows = len(theta_matrix_info.item_id)
        num_cols = theta_matrix_info.topics_count
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)

        cp_args = messages.CopyRequestResultArgs()
        cp_args.request_type = constants.CopyRequestResultArgs_RequestType_GetThetaSecondPass
        self._lib.ArtmCopyRequestResultEx(numpy_ndarray, cp_args)

        return numpy_ndarray

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

        result = self._lib.ArtmRequestTopicModel(self.master_id, args)

        phi_matrix_info = messages.TopicModel()
        phi_matrix_info.ParseFromString(result)

        return phi_matrix_info

    def get_phi_matrix(self, model, topic_names=None):
        """Args:
           - model(str): name of matrix in BigARTM
           - topic_names(list of str): list of topics to retrieve (None == all topics)
           Returns:
           - numpy.ndarray with Phi data (e.g. p(w|t) values)
        """
        args = messages.GetTopicModelArgs()
        args.model_name = model
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)

        result = self._lib.ArtmRequestTopicModelExternal(self.master_id, args)

        phi_matrix_info = messages.TopicModel()
        phi_matrix_info.ParseFromString(result)

        num_rows = len(phi_matrix_info.token)
        num_cols = phi_matrix_info.topics_count
        numpy_ndarray = numpy.zeros(shape=(num_rows, num_cols), dtype=numpy.float32)

        cp_args = messages.CopyRequestResultArgs()
        cp_args.request_type = constants.CopyRequestResultArgs_RequestType_GetModelSecondPass
        self._lib.ArtmCopyRequestResultEx(numpy_ndarray, cp_args)

        return numpy_ndarray

    def export_model(self, model, file_name):
        args = messages.ExportModelArgs()
        args.model_name = model
        args.file_name = file_name

        result = self._lib.ArtmExportModel(self.master_id, args)

    def import_model(self, model, file_name):
        """Args:
           - model(str): name of matrix in BigARTM
           - file_name(str): the name of file to load model from binary format
        """
        args = messages.ImportModelArgs()
        args.model_name = model
        args.file_name = file_name

        result = self._lib.ArtmImportModel(self.master_id, args)
