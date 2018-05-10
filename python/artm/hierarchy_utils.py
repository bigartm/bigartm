# Copyright 2017, Additive Regularization of Topic Models.

import artm
import uuid
import copy
import numpy as np
import pandas
import os.path
import os
import pickle
import glob
import warnings

from six.moves import range


class hARTM(object):

    def __init__(self, num_processors=None, class_ids=None,
                 scores=None, regularizers=None, num_document_passes=10, reuse_theta=False,
                 dictionary=None, cache_theta=False, theta_columns_naming='id', seed=-1,
                 tmp_files_path=""):
        """
        :Description: a class for constructing topic hierarchy that is
                      a sequence of tied artm.ARTM() models (levels)

        :param int num_processors: how many threads will be used for model training, if\
                                 not specified then number of threads will be detected by the lib
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
        :param str tmp_files_path: a path where to save temporary files (temporary solution),
                                   default value: current directory

        :Usage:
          *  to construct hierarchy you have to learn several ARTM models:
                 hier = artm.hARTM()
                 level0 = hier.add_level(num_topics=5)   # returns artm.ARTM() instance
                 # work with level0 as with usual model
                 level1 = hier.add_level(num_topics=25, parent_level_weight=1)
                 # work with level1 as with usual model
                 # ...
          *  to get the i-th level's model, use
                  level = hier[i]
              or
                  level = hier.get_level(i)
          * to iterate through levels use
                  for level in hier:
                      # some work with level
          * method hier.del_level(...) removes i-th level and all levels after it
          * other hARTM methods correspond to those in ARTM class and call them
            sequantially for all levels of hierarchy from 0 to the last one.
            For example, to fit levels offline you may call fit_offline method
            of hARTM instance or of each level individually.
        """
        self._common_models_args = {"num_processors": num_processors,
                                    "class_ids": class_ids,
                                    "scores": scores,
                                    "regularizers": regularizers,
                                    "num_document_passes": num_document_passes,
                                    "reuse_theta": reuse_theta,
                                    "dictionary": dictionary,
                                    "cache_theta": cache_theta,
                                    "theta_columns_naming": theta_columns_naming}
        if seed > 0:
            self._seed = seed
        else:
            self._seed = 321
        self._tmp_files_path = tmp_files_path
        self._model_name = "n_wt"   # now it is not user feature
        self._levels = []

    # ========== PROPERTIES ==========
    @property
    def num_processors(self):
        return self._common_models_args["num_processors"]

    @property
    def cache_theta(self):
        return self._common_models_args["cache_theta"]

    @property
    def reuse_theta(self):
        return self._common_models_args["reuse_theta"]

    @property
    def num_document_passes(self):
        return self._common_models_args["num_document_passes"]

    @property
    def theta_columns_naming(self):
        return self._common_models_args["theta_columns_naming"]

    @property
    def class_ids(self):
        return self._common_models_args["class_ids"]

    @property
    def regularizers(self):
        return self._common_models_args["regularizers"]

    @property
    def scores(self):
        return self._common_models_args["scores"]

    @property
    def dictionary(self):
        return self._common_models_args["dictionary"]

    @property
    def seed(self):
        return self._seed

    @property
    def library_version(self):
        """
        :Description: the version of BigARTM library in a MAJOR.MINOR.PATCH format
        """
        return self._lib.version()

    @property
    def num_levels(self):
        return len(self._levels)

    @property
    def tmp_files_path(self):
        return self._tmp_files_path

    # ========== SETTERS ==========
    @num_processors.setter
    def num_processors(self, num_processors):
        if num_processors <= 0 or not isinstance(num_processors, int):
            raise IOError('Number of processors should be a positive integer')
        else:
            for level in self._levels:
                level.num_processors = num_processors
            self._common_models_args["num_processors"] = num_processors

    @cache_theta.setter
    def cache_theta(self, cache_theta):
        if not isinstance(cache_theta, bool):
            raise IOError('cache_theta should be bool')
        else:
            for level in self._levels:
                level.cache_theta = cache_theta
            self._common_models_args["cache_theta"] = cache_theta

    @reuse_theta.setter
    def reuse_theta(self, reuse_theta):
        if not isinstance(reuse_theta, bool):
            raise IOError('reuse_theta should be bool')
        else:
            for level in self._levels:
                level.reuse_theta = reuse_theta
            self._common_models_args["reuse_theta"] = reuse_theta

    @num_document_passes.setter
    def num_document_passes(self, num_document_passes):
        if num_document_passes <= 0 or not isinstance(num_document_passes, int):
            raise IOError(
                'Number of passes through document should be a positive integer')
        else:
            for level in self._levels:
                level.num_document_passes = num_document_passes
            self._common_models_args[
                "num_document_passes"] = num_document_passes

    @theta_columns_naming.setter
    def theta_columns_naming(self, theta_columns_naming):
        if theta_columns_naming not in ['id', 'title']:
            raise IOError(
                'theta_columns_naming should be either "id" or "title"')
        else:
            for level in self._levels:
                level.theta_columns_naming = theta_columns_naming
            self._common_models_args[
                "theta_columns_naming"] = theta_columns_naming

    @class_ids.setter
    def class_ids(self, class_ids):
        if len(class_ids) < 0:
            raise IOError(
                'Number of (class_id, class_weight) pairs should be non-negative')
        else:
            for level in self._levels:
                level.class_ids = class_ids
            self._common_models_args["class_ids"] = class_ids

    @scores.setter
    def scores(self, scores):
        if not isinstance(scores, list):
            raise IOError('scores should be a list')
        else:

            self._common_models_args["scores"] = scores

    @regularizers.setter
    def regularizers(self, regularizers):
        if not isinstance(regularizers, list):
            raise IOError('scores should be a list')
        else:
            self._common_models_args["regularizers"] = regularizers

    @dictionary.setter
    def dictionary(self, dictionary):
        self._common_models_args["dictionary"] = dictionary

    @seed.setter
    def seed(self, seed):
        if seed < 0 or not isinstance(seed, int):
            raise IOError('Random seed should be a positive integer')
        else:
            self._seed = seed

    @tmp_files_path.setter
    def tmp_files_path(self, tmp_files_path):
        self._tmp_files_path = tmp_files_path

    # ========== METHODS ==========
    def _get_seed(self, level_idx):
        np.random.seed(self._seed)
        return int(np.random.randint(10000, size=level_idx + 1)[-1])

    def add_level(self, num_topics=None, topic_names=None, parent_level_weight=1):
        """
        :Description: adds new level to the hierarchy

        :param int num_topics: the number of topics in level model, will be overwriten if
                               parameter topic_names is set
        :param topic_names: names of topics in model
        :type topic_names: list of str
        :param float parent_level_weight: the coefficient of smoothing n_wt by n_wa,
                                          a enumerates parent topics

        :return: ARTM or derived ARTM_Level instance

        :Notes:
          *  hierarchy structure assumes the number of topics on each following level is greater
             than on previous one
          *  work with returned value as with usual ARTM model
          *  to access any level, use [] or get_level method
          *  Important! You cannot add next level before previous one is initialized and fit.
        """
        if len(self._levels) and num_topics <= self._levels[-1].num_topics:
            warnings.warn("Adding level with num_topics = %s less or equal than parent level's num_topics = %s" %
                          (num_topics, self._levels[-1].num_topics))
        level_idx = len(self._levels)
        if not len(self._levels):
            self._levels.append(artm.ARTM(num_topics=num_topics,
                                          topic_names=topic_names,
                                          seed=self._get_seed(level_idx),
                                          **self._common_models_args))
        else:
            self._levels.append(ARTM_Level(parent_model=self._levels[-1],
                                           phi_batch_weight=parent_level_weight,
                                           phi_batch_path=self._tmp_files_path,
                                           model_name=self._model_name,
                                           num_topics=num_topics,
                                           topic_names=topic_names,
                                           seed=self._get_seed(level_idx),
                                           **self._common_models_args))
        level = self._levels[-1]
        config = level.master._config
        config.opt_for_avx = False
        level.master._lib.ArtmReconfigureMasterModel(
            level.master.master_id, config)
        return level

    def del_level(self, level_idx):
        """
        :Description: removes i-th level and all following levels.

        :param int level_idx: the number of level from what to start removing
                      if -1, the last level is removed
        """
        if level_idx == -1:
            del self._levels[-1]
            return
        for _ in range(level_idx, len(self._levels)):
            del self._levels[-1]

    def get_level(self, level_idx):
        """
        :Description: access level

        :param int level_idx: the number of level to return

        :return: specified level that is ARTM or derived ARTM_Level instance
        """
        return self._levels[level_idx]

    def __getitem__(self, i):
        return self._levels[i]

    def __iter__(self):
        return iter(self._levels)

    def __enter__(self):
        return self

    def dispose(self):
        """
        :Description: free all native memory, allocated for this hierarchy

        :Note:
          * This method does not free memory occupied by models' dictionaries,
            because dictionaries are shared across all models
          * hARTM class implements __exit__ and __del__ methods,
            which automatically call dispose.
        """
        for level in self._levels:
            level.dispose()
        self._levels = []

    def __exit__(self, exc_type, exc_value, traceback):
        self.dispose()

    def __del__(self):
        self.dispose()

    def fit_offline(self, batch_vectorizer, num_collection_passes=1):
        """
        :Description: proceeds the learning of all hirarchy levels
                      from 0 to the last one

        :param object_referenece batch_vectorizer: an instance of BatchVectorizer class
        :param int num_collection_passes: number of iterations over whole given collection
                                          for each level

        :Note:
          * You cannot add add next level before previous one is fit.
            So use this method only when all levels are added, initialized and fit,
            for example, when you added one more regularizer or loaded hierarchy from disk.

        """
        for level in self._levels:
            level.fit_offline(batch_vectorizer, num_collection_passes)

    def save(self, path):
        """
        :Description: saves all levels

        :param str path: a path where to save hierarchy files
                         This must be existing empty path,
                         otherwise exception is raised
        """
        if len(glob.glob(os.path.join(path, "*"))):
            raise ValueError("Passed path should be empty")
        for level_idx, level in enumerate(self._levels):
            level.save(os.path.join(path, "level" +
                                    str(level_idx) + "_nwt.model"), model_name="n_wt")
            level.save(os.path.join(path, "level" +
                                    str(level_idx) + "_pwt.model"), model_name="p_wt")
        info = {"parent_level_weight": [
            level.parent_level_weight for level in self._levels[1:]]}
        with open(os.path.join(path, "info.dump"), "wb") as fout:
            pickle.dump(info, fout)

    def load(self, path):
        """
        :Description: loads models of already constructed hierarchy

        :param str path: a path where hierarchy was saved by hARTM.save method

        :Notes:
          * Loaded models will overwrite ARTM.topic_names and class_ids fields of each level.
          * All class_ids weights will be set to 1.0, you need to specify them by\
            hand if it's necessary.
          * The method call will empty ARTM.score_tracker of each level.
          * All regularizers and scores will be forgotten.
          * etc.
          * We strongly recommend you to reset all important parameters of the ARTM\
            models and hARTM, used earlier.
        """
        info_filename = glob.glob(os.path.join(path, "info.dump"))
        if len(info_filename) != 1:
            raise ValueError("Given path is not hARTM safe")
        with open(info_filename[0], "rb") as fin:
            info = pickle.load(fin)
        model_filenames = glob.glob(os.path.join(path, "*.model"))
        if len({len(info["parent_level_weight"]) + 1, len(model_filenames) / 2}) > 1:
            raise ValueError("Given path is not hARTM safe")
        self._levels = []
        sorted_model_filenames = sorted(model_filenames)
        for level_idx in range(len(model_filenames) // 2):
            if not len(self._levels):
                model = artm.ARTM(num_topics=1,
                                  seed=self._get_seed(level_idx),
                                  **self._common_models_args)
            else:
                parent_level_weight = info[
                    "parent_level_weight"][level_idx - 1]
                model = ARTM_Level(parent_model=self._levels[-1],
                                   phi_batch_weight=parent_level_weight,
                                   phi_batch_path=self._tmp_files_path,
                                   num_topics=1,
                                   seed=self._get_seed(level_idx),
                                   **self._common_models_args)
            filename = sorted_model_filenames[2 * level_idx + 1]
            model.load(filename, "p_wt")
            filename = sorted_model_filenames[2 * level_idx]
            model.load(filename, "n_wt")
            config = model.master._config
            config.opt_for_avx = False
            model.master._lib.ArtmReconfigureMasterModel(
                model.master.master_id, config)
            self._levels.append(model)

    def get_theta(self, topic_names=None):
        """
        :Description: get level-wise vertically stacked Theta matrices
                      for training set of documents

        :param topic_names: list with topics to extract, None means all topics
        :type topic_names: list of str

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the ids of documents, for which the Theta matrix was requested;
          * rows --- the names of topics in format level_X_Y
                     where X is level index and Y is topic name;
          * data --- content of Theta matrix.
        """
        if not len(self._levels):
            return
        theta_0_pd = self._levels[0].get_theta()
        columns = theta_0_pd.columns
        index = "level_0_" + theta_0_pd.index
        theta = theta_0_pd.values
        for level_idx, level in enumerate(self._levels[1:]):
            theta_pd = level.get_theta()
            theta = np.vstack((theta, theta_pd.values))
            index = index.append(
                "level" + str(level_idx + 1) + "_" + theta_pd.index)
        return pandas.DataFrame(data=theta, columns=columns, index=index)

    def transform(self, batch_vectorizer):
        """
        :Description: get level-wise vertically stacked Theta matrices
                      for new documents

        :param object_reference batch_vectorizer: an instance of BatchVectorizer class

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the ids of documents, for which the Theta matrix was requested;
          * rows --- the names of topics in format level_X_Y
                     where X is level index and Y is topic name;
          * data --- content of Theta matrix.

        :Note:
          * to access p(t|d, w) matrix or to predict class use transform method
            of hierarchy level individually
        """
        if not len(self._levels):
            return
        theta_0_pd = self._levels[0].transform(batch_vectorizer)
        columns = theta_0_pd.columns
        index = "level_0_" + theta_0_pd.index
        theta = theta_0_pd.values
        for level_idx, level in enumerate(self._levels[1:]):
            theta_pd = level.transform(batch_vectorizer)
            theta = np.vstack((theta, theta_pd.values))
            index = index.append(
                "level" + str(level_idx + 1) + "_" + theta_pd.index)
        return pandas.DataFrame(data=theta, columns=columns, index=index)

    def get_phi(self, class_ids=None, model_name=None):
        """
        :Description: get level-wise horizontally stacked Phi matrices

        :param class_ids: list with class_ids or single class_id to extract, None means all class ids
        :type class_ids: list of str or str
        :param str model_name: self.model_pwt by default, self.model_nwt is also\
                      reasonable to extract unnormalized counters

        :return:
          * pandas.DataFrame: (data, columns, rows), where:
          * columns --- the names of topics in format level_X_Y
                     where X is level index and Y is topic name;
          * rows --- the tokens of topic model;
          * data --- content of Phi matrix.

        :Note:
          * if you need to extract specified topics, use get_phi() method of individual level model
        """
        phi = pandas.concat([level.get_phi(class_ids=class_ids, model_name=model_name)
                             for level in self._levels], axis=1)
        phi.columns = pandas.Series(["level" + str(level_idx) for level_idx, level in enumerate(self._levels)
                                     for i in range(level.num_topics)]).\
            str.cat(phi.columns, sep="_")
        return phi

    def clone(self):
        """
        :Description: returns a deep copy of the artm.hARTM object

        :Note:
          * This method is equivalent to copy.deepcopy() of your artm.hARTM object.
            For more information refer to artm.ARTM.clone() method.
        """
        return copy.deepcopy(self)


class ARTM_Level(artm.ARTM):

    def __init__(self, parent_model, phi_batch_weight=1, phi_batch_path=".",
                 model_name="n_wt", *args, **kwargs):
        """
        :description: builds one hierarchy level that is usual topic model

        :param parent_model: ARTM or ARTM_Level instance, previous level model,
         already built
        :param phi_batch_weight: float, weight of parent phi batch
        :param phi_batch_path: string, path where to save phi parent batch,
         default '', temporary solution
        :other params as in ARTM class
        :param model_nwt: string, "n_wt" or "p_wt", which model to use in parent_batch

        :Notes:
             * Parent phi batch consists of documents that are topics (phi columns)
               of parent_model. Corresponding to this batch Theta part is psi matrix
               with p(subtopic|topic) elements.
             * To get psi matrix use get_psi() method.
               Other methods are as in ARTM class.
        """
        if model_name not in {"n_wt", "p_wt"}:
            raise ValueError(
                "Parameter model_name should be either 'n_wt' or 'p_wt'")
        self._parent_model = parent_model
        self._phi_batch_weight = phi_batch_weight
        self._level = 1 if "_level" not in dir(
            parent_model) else parent_model._level + 1
        self._name = "level" + str(self._level)
        self._phi_batch_path = os.path.join(
            phi_batch_path, "phi" + str(self._level) + ".batch")
        self._model_name = model_name
        super(ARTM_Level, self).__init__(*args, **kwargs)
        self._create_parent_phi_batch()
        self._create_parent_phi_batch_vectorizer()

    @property
    def parent_level_weight(self):
        return self._phi_batch_weight

    @parent_level_weight.setter
    def parent_level_weight(self, parent_level_weight):
        self._phi_batch_weight = parent_level_weight

    def _create_parent_phi_batch(self):
        """
        :description: creates new batch with parent level topics as documents
        """
        batch_dict = {}
        NNZ = 0
        batch = artm.messages.Batch()
        batch.id = str(uuid.uuid4())
        batch_id = batch.id
        batch.description = "__parent_phi_matrix_batch__"
        for topic_name_idx in range(len(self._parent_model.topic_names)):
            topic_name = self._parent_model.topic_names[topic_name_idx]
            if self._model_name == "p_wt":
                phi = self._parent_model.get_phi(
                    topic_names={topic_name}, model_name=self._parent_model.model_pwt)
            else:
                phi = self._parent_model.get_phi(
                    topic_names={topic_name}, model_name=self._parent_model.model_nwt)
            if not topic_name_idx:
                # batch.token is the same for all topics, create it once
                topics_and_tokens_info = \
                    self._parent_model.master.get_phi_info(
                        self._parent_model.model_nwt)
                for token, class_id in \
                        zip(topics_and_tokens_info.token, topics_and_tokens_info.class_id):
                    if token not in batch_dict:
                        batch.token.append(token)
                        batch.class_id.append(class_id)
                        batch_dict[token] = len(batch.token) - 1

            # add item (topic) to batch
            item = batch.item.add()
            item.title = topic_name
            field = item.field.add()
            indices = phi[topic_name] > 0
            for token, weight in  \
                    zip(phi.index[indices], phi[topic_name][indices]):
                field.token_id.append(batch_dict[token])
                field.token_weight.append(float(weight))
                NNZ += weight
        self.parent_batch = batch
        with open(self._phi_batch_path, 'wb') as fout:
            fout.write(batch.SerializeToString())

    def _create_parent_phi_batch_vectorizer(self):
        self.phi_batch_vectorizer = artm.BatchVectorizer(batches=[''])
        self.phi_batch_vectorizer._batches_list[0] = artm.batches_utils.Batch(
            self._phi_batch_path)
        self.phi_batch_vectorizer._weights = [self._phi_batch_weight]

    def fit_offline(self, batch_vectorizer, num_collection_passes=1, *args, **kwargs):
        modified_batch_vectorizer = artm.BatchVectorizer(batches=[''], data_path=batch_vectorizer.data_path,
                                                         batch_size=batch_vectorizer.batch_size,
                                                         gather_dictionary=False)

        del modified_batch_vectorizer.batches_list[0]
        del modified_batch_vectorizer.weights[0]
        for batch, weight in zip(batch_vectorizer.batches_list, batch_vectorizer.weights):
            modified_batch_vectorizer.batches_list.append(batch)
            modified_batch_vectorizer.weights.append(weight)
        modified_batch_vectorizer.batches_list.append(
            artm.batches_utils.Batch(self._phi_batch_path))
        modified_batch_vectorizer.weights.append(self._phi_batch_weight)
        # import_batches_args = artm.wrapper.messages_pb2.ImportBatchesArgs(
        #                           batch=[self.parent_batch])
        # self._lib.ArtmImportBatches(self.master.master_id, import_batches_args)

        super(ARTM_Level, self).fit_offline(modified_batch_vectorizer, num_collection_passes=num_collection_passes,
                                            *args, **kwargs)

    def fit_online(self, *args, **kwargs):
        raise NotImplementedError(
            "Fit_online method is not implemented in hARTM. Use fit_offline method.")

    def get_psi(self):
        """
        :returns: p(subtopic|topic) matrix
        """
        current_columns_naming = self.theta_columns_naming
        self.theta_columns_naming = "title"
        psi = self.transform(self.phi_batch_vectorizer)
        self.theta_columns_naming = current_columns_naming
        return psi

    def get_theta(self, topic_names=None):
        theta_info = self.master.get_theta_info()

        all_topic_names = [topic_name for topic_name in theta_info.topic_name]
        use_topic_names = topic_names if topic_names is not None else all_topic_names
        _, nd_array = self.master.get_theta_matrix(topic_names=use_topic_names)

        titles_list = [item_title for item_title in theta_info.item_title]
        theta_data_frame = pandas.DataFrame(data=nd_array.transpose(),
                                            columns=titles_list,
                                            index=use_topic_names)
        item_idxs = np.logical_not(
            theta_data_frame.columns.isin(self._parent_model.topic_names))
        theta_data_frame = theta_data_frame.drop(
            self._parent_model.topic_names, axis=1)
        if self._theta_columns_naming == "id":
            ids_list = [item_id for item_id in theta_info.item_id]
            theta_data_frame.columns = np.array(ids_list)[item_idxs]
        return theta_data_frame

    def clone(self):
        """
        :Description: returns a deep copy of the artm.ARTM_Level object

        :Note:
          * This method is equivalent to copy.deepcopy() of your artm.ARTM_Level object.
            For more information refer to artm.ARTM.clone() method.
        """
        return copy.deepcopy(self)
