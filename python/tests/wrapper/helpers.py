import os

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants


class TestHelper(object):
    def __init__(self, library):
        self._lib = library
        self.master_id = None

    def create_master_component(self, num_processors=None, scores=None):
        """ Args:
            - num_processors (int): number of work threads to use
            - scores (list): list of tuples (name, config) for each config to use
            Returns:
            (int): the id of created master
        """
        master_config = messages.MasterComponentConfig()

        if num_processors is not None:
            master_config.processors_count = num_processors
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

        return self._lib.ArtmCreateMasterComponent(master_config)

    def parse_collection_uci(self, docword, vocab, target_folder, dictionary_name):
        """Args:
           - docword(str): full path to docword.<name>.txt
           - vocab(str): full path to vocab.<name>.txt
           - target_folder(str): full path to folder for batches
           - dictionry_name(str): the name of future collection dictionary
        """
        config = messages.CollectionParserConfig()
        config.format = constants.CollectionParserConfig_Format_BagOfWordsUci

        config.docword_file_path = docword
        config.vocab_file_path = vocab
        config.target_folder = target_folder
        config.dictionary_file_name = dictionary_name

        self._lib.ArtmParseCollection(config)

    def import_dictionary(self, file_name, dictionary_name, master_id=None):
        """Args:
           - file_name(str): full name of dictionary file
           - dictionary_name(str): name of imported dictionary
           - master_id(int): the id of master component returned by create_master_component()
        """
        args = messages.ImportDictionaryArgs()
        args.dictionary_name = dictionary_name
        args.file_name = file_name
        
        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        self._lib.ArtmImportDictionary(cur_master_id, args)

    def initialize_model(self, model_name, num_topics, source_type=None,
                         disk_path=None, dictionary_name=None, master_id=None):
        """Args:
           - model_name(str): name of pwt matrix in BigARTM
           - source_type(str): 'batches' | 'dictionary'
           - num_topics(int): number of topics in model
           - disk_path(str): full name of folder with batches (need if InitializeModelArgs_SourceType_Batches)
           - dictionary_name(str): name of imported dictionary (need if InitializeModelArgs_SourceType_Dictionary)
           - master_id(int): the id of master component returned by create_master_component()
        """
        args = messages.InitializeModelArgs()
        args.model_name = model_name
        args.topics_count = num_topics

        if source_type is 'batches':
            args.source_type = constants.InitializeModelArgs_SourceType_Batches
            if disk_path is not None:
                args.disk_path = disk_path
        elif source_type is 'dictionary':
            args.source_type = constants.InitializeModelArgs_SourceType_Dictionary
            if dictionary_name is not None:
                args.dictionary_name = dictionary_name

        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        self._lib.ArtmInitializeModel(cur_master_id, args)

    def retrieve_score(self, model_name, score_name, master_id=None):
        """Args:
           - model_name(str): name of pwt matrix in BigARTM
           - score_name(str): the user defined name of score to retrieve
           - score_config: reference to score data object
           - master_id(int): the id of master component returned by create_master_component()
        """
        args = messages.GetScoreValueArgs()
        args.model_name = model_name
        args.score_name = score_name

        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        results = self._lib.ArtmRequestScore(cur_master_id, args)
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

    def process_batches(self, pwt, nwt, num_inner_iterations, batches_folder,
                        regularizer_name=None, regularizer_tau=None,
                        class_ids=None, class_weights=None, master_id=None):
        """Args:
           - pwt(str): name of pwt matrix in BigARTM
           - nwt(str): name of nwt matrix in BigARTM
           - num_inner_iterations(int): number of inner iterations during processing
           - batches_folder(str): full path to data folder
           - regularizer_name(list of str): list of names of Theta regularizers to use
           - regularizer_tau(list of double): list of tau coefficients for Theta regularizers
           - class_ids(list of str): list of class ids to use during processing
           - class_weights(list of double): list of corresponding weights of class ids
           - master_id(int): the id of master component returned by create_master_component()
        """
        args = messages.ProcessBatchesArgs()
        args.pwt_source_name = pwt
        args.nwt_target_name = nwt
        for name in os.listdir(batches_folder):
            _, extension = os.path.splitext(name)
            if extension == '.batch':
                args.batch_filename.append(os.path.join(batches_folder, name))
        args.inner_iterations_count = num_inner_iterations

        if regularizer_name is not None and regularizer_tau is not None:
            for name, tau in zip(regularizer_name, regularizer_tau):
                args.regularizer_name.append(name)
                args.regularizer_tau.append(tau)

        if class_ids is not None and class_weights is not None:
            for class_id, weight in zip(class_ids, class_weights):
                args.class_id.append(class_id)
                args.class_weight.append(weight)

        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        self._lib.ArtmRequestProcessBatches(cur_master_id, args)

    def regularize_model(self, pwt, nwt, rwt, regularizer_name, regularizer_tau, master_id=None):
        """Args:
           - pwt(str): name of pwt matrix in BigARTM
           - nwt(str): name of nwt matrix in BigARTM
           - rwt(str): name of rwt matrix in BigARTM
           - regularizer_name(list of str): list of names of Phi regularizers to use
           - regularizer_tau(list of double): list of tau coefficients for Phi regularizers
           - master_id(int): the id of master component returned by create_master_component()
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

        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        self._lib.ArtmRegularizeModel(cur_master_id, args)

    def normalize_model(self, pwt, nwt, rwt=None, master_id=None):
        """Args:
           - pwt(str): name of pwt matrix in BigARTM
           - nwt(str): name of nwt matrix in BigARTM
           - rwt(str): name of rwt matrix in BigARTM
           - master_id(int): the id of master component returned by create_master_component()
        """
        args = messages.NormalizeModelArgs()
        args.pwt_target_name = pwt
        args.nwt_source_name = nwt
        if rwt is not None:
            args.rwt_source_name = rwt

        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        self._lib.ArtmNormalizeModel(cur_master_id, args)

    def create_smooth_sparse_phi_regularizer(self, name, class_ids=None, dictionary_name=None, master_id=None):
        """Args:
           - name(str): the name of the future regularizer
           - class_ids(list of str): the list of class_ids to be regularized
           - dictionary_name(str): name of imported dictionary
           - master_id(int): the id of master component returned by create_master_component()
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

        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        ref_reg_config.config = config.SerializeToString()
        self._lib.ArtmCreateRegularizer(cur_master_id, ref_reg_config)

    def create_smooth_sparse_theta_regularizer(self, name, master_id=None):
        """Args:
           - name(str): the name of the future regularizer
           - master_id(int): the id of master component returned by create_master_component()
        """
        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = name
        ref_reg_config.type = constants.RegularizerConfig_Type_SmoothSparseTheta
        ref_reg_config.config = messages.SmoothSparseThetaConfig().SerializeToString()

        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        self._lib.ArtmCreateRegularizer(cur_master_id, ref_reg_config)

    def create_decorrelator_phi_regularizer(self, name, class_ids=None, master_id=None):
        """Args:
           - name(str): the name of the future regularizer
           - class_ids(list of str): the list of class_ids to be regularized
           - master_id(int): the id of master component returned by create_master_component()
        """
        ref_reg_config = messages.RegularizerConfig()
        ref_reg_config.name = name
        ref_reg_config.type = constants.RegularizerConfig_Type_DecorrelatorPhi
        
        config = messages.DecorrelatorPhiConfig()
        if class_ids is not None:
            for class_id in class_ids:
                config.class_id.append(class_id)
        
        ref_reg_config.config = config.SerializeToString()
        cur_master_id = self.master_id
        if master_id is not None:
            cur_master_id = master_id
        self._lib.ArtmCreateRegularizer(cur_master_id, ref_reg_config)
