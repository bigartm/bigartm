import collections


__all__ = [
    'SparsityPhiScoreTracker',
    'ItemsProcessedScoreTracker',
    'PerplexityScoreTracker',
    'SparsityThetaScoreTracker',
    'ThetaSnippetScoreTracker',
    'TopicKernelScoreTracker',
    'TopTokensScoreTracker',
    'TopicMassPhiScoreTracker'
]


class SparsityPhiScoreTracker(object):
    """SparsityPhiScoreTracker represents a result of counting
    SparsityPhiScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._zero_tokens = []
        self._total_tokens = []

    def add(self, score=None):
        """SparsityPhiScoreTracker.add() --- add info about score after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.master.retrieve_score(score._model, score.name)

            self._value.append(_data.value)
            self._zero_tokens.append(_data.zero_tokens)
            self._total_tokens.append(_data.total_tokens)
        else:
            self._value.append(None)
            self._zero_tokens.append(None)
            self._total_tokens.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of double: value of Phi sparsity on synchronizations
        """
        return self._value

    @property
    def zero_tokens(self):
        """Returns:
          list of int: number of zero rows in Phi on synchronizations
        """
        return self._zero_tokens

    @property
    def total_tokens(self):
        """Returns:
          list of int: total number of rows in Phi on synchronizations
        """
        return self._total_tokens

    @property
    def last_value(self):
        """Returns:
        double: value of Phi sparsity on the last synchronization
        """
        return self._value[-1]

    @property
    def last_zero_tokens(self):
        """Returns:
        int: number of zero rows in Phi on the last synchronization
        """
        return self._zero_tokens[-1]

    @property
    def last_total_tokens(self):
        """Returns:
        int: total number of rows in Phi on the last synchronization
        """
        return self._total_tokens[-1]


###################################################################################################
class SparsityThetaScoreTracker(object):
    """SparsityThetaScoreTracker represents a result of counting
    SparsityThetaScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._zero_topics = []
        self._total_topics = []

    def add(self, score=None):
        """SparsityThetaScoreTracker.add() --- add info about score
        after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.master.retrieve_score(score._model, score.name)

            self._value.append(_data.value)
            self._zero_topics.append(_data.zero_topics)
            self._total_topics.append(_data.total_topics)
        else:
            self._value.append(None)
            self._zero_topics.append(None)
            self._total_topics.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of double: value of Theta sparsity on synchronizations
        """
        return self._value

    @property
    def zero_topics(self):
        """Returns:
          list of int: number of zero rows in Theta on synchronizations
        """
        return self._zero_topics

    @property
    def total_topics(self):
        """Returns:
          list of int: total number of rows in Theta on synchronizations
        """
        return self._total_topics

    @property
    def last_value(self):
        """Returns:
          double: value of Theta sparsity on the last synchronization
        """
        return self._value[-1]

    @property
    def last_zero_topics(self):
        """Returns:
          int: number of zero rows in Theta on the last synchronization
        """
        return self._zero_topics[-1]

    @property
    def last_total_topics(self):
        """Returns:
          int: total number of rows in Theta on the last synchronization
        """
        return self._total_topics[-1]


###################################################################################################
class PerplexityScoreTracker(object):
    """PerplexityScoreTracker represents a result of counting PerplexityScore
    (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._raw = []
        self._normalizer = []
        self._zero_tokens = []
        self._theta_sparsity_value = []
        self._theta_sparsity_zero_topics = []
        self._theta_sparsity_total_topics = []

    def add(self, score=None):
        """PerplexityScoreTracker.add() --- add info about score after
        synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.master.retrieve_score(score._model, score.name)

            self._value.append(_data.value)
            self._raw.append(_data.raw)
            self._normalizer.append(_data.normalizer)
            self._zero_tokens.append(_data.zero_words)
            self._theta_sparsity_value.append(_data.theta_sparsity_value)
            self._theta_sparsity_zero_topics.append(_data.theta_sparsity_zero_topics)
            self._theta_sparsity_total_topics.append(_data.theta_sparsity_total_topics)
        else:
            self._value.append(None)
            self._raw.append(None)
            self._normalizer.append(None)
            self._zero_tokens.append(None)
            self._theta_sparsity_value.append(None)
            self._theta_sparsity_zero_topics.append(None)
            self._theta_sparsity_total_topics.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of double: value of perplexity on synchronizations
        """
        return self._value

    @property
    def raw(self):
        """Returns:
          list of double: raw value in formula of perplexity on synchronizations
        """
        return self._raw

    @property
    def normalizer(self):
        """Returns:
          list double: normalizer value in formula of perplexity on synchronizations
        """
        return self._normalizer

    @property
    def zero_tokens(self):
        """Returns:
          list of int: number of tokens with zero counters on synchronizations
        """
        return self._zero_tokens

    @property
    def theta_sparsity_value(self):
        """Returns:
          list of double: Theta sparsity value on synchronizations
        """
        return self._theta_sparsity_value

    @property
    def theta_sparsity_zero_topics(self):
        """Returns:
        list of int: number of zero rows in Theta on synchronizations
        """
        return self._theta_sparsity_zero_topics

    @property
    def theta_sparsity_total_topics(self):
        """Returns:
          list of int: total number of rows in Theta on synchronizations
        """
        return self._theta_sparsity_total_topics

    @property
    def last_value(self):
        """Returns:
          double: value of perplexity on the last synchronization
        """
        return self._value[-1]

    @property
    def last_raw(self):
        """Returns:
          double: raw value in formula of perplexity on the last synchronization
        """
        return self._raw[-1]

    @property
    def last_normalizer(self):
        """Returns:
          double: normalizer value in formula of perplexity on the last synchronization
        """
        return self._normalizer[-1]

    @property
    def last_zero_tokens(self):
        """Returns:
          int: number of tokens with zero counters on the last synchronization
        """
        return self._zero_tokens[-1]

    @property
    def last_theta_sparsity_value(self):
        """Returns:
          double: Theta sparsity value on the last synchronization
        """
        return self._theta_sparsity_value[-1]

    @property
    def last_theta_sparsity_zero_topics(self):
        """Returns:
          int: number of zero rows in Theta on the last synchronization
        """
        return self._theta_sparsity_zero_topics[-1]

    @property
    def last_theta_sparsity_total_topics(self):
        """Returns:
          int: total number of rows in Theta on the last synchronization
        """
        return self._theta_sparsity_total_topics[-1]


###################################################################################################
class ItemsProcessedScoreTracker(object):
    """ItemsProcessedScoreTracker represents a result of counting
    ItemsProcessedScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []

    def add(self, score=None):
        """ItemsProcessedScoreTracker.add() --- add info about score
        after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.master.retrieve_score(score._model, score.name)
            self._value.append(_data.value)
        else:
            self._value.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of int: total number of processed documents on synchronizations
        """
        return self._value

    @property
    def last_value(self):
        """Returns:
          int: total number of processed documents on the last synchronization
        """
        return self._value[-1]


###################################################################################################
class TopTokensScoreTracker(object):
    """TopTokensScoreTracker represents a result of counting TopTokensScore
    (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._num_tokens = []
        self._topic_info = []
        self._average_coherence = []

    def add(self, score=None):
        """TopTokensScoreTracker.add() --- add info about score
        after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.master.retrieve_score(score._model, score.name)

            self._num_tokens.append(_data.num_entries)

            self._topic_info.append({})
            for top_idx, top_name in enumerate(_data.topic_name):
                tokens = []
                weights = []
                for i in xrange(_data.num_entries):
                    if _data.topic_name[i] == top_name:
                        tokens.append(_data.token[i])
                        weights.append(_data.weight[i])
                coherence = -1
                if len(_data.coherence.value) > 0:
                    coherence = _data.coherence.value[top_idx]
                self._topic_info[-1][top_name] = \
                    collections.namedtuple('TopTokensScoreTuple',
                                           ['tokens', 'weights', 'coherence'])
                self._topic_info[-1][top_name].tokens = tokens
                self._topic_info[-1][top_name].weights = weights
                self._topic_info[-1][top_name].coherence = coherence

            self._average_coherence.append(_data.average_coherence)
        else:
            self._num_tokens.append(None)
            self._topic_info.append(None)
            self._average_coherence.append(None)

    @property
    def name(self):
        return self._name

    @property
    def num_tokens(self):
        """Returns:
          list of int: reqested number of top tokens in each topic on
        synchronizations
        """
        return self._num_tokens

    @property
    def topic_info(self):
        """Returns:
          list of sets: information about top tokens per topic on synchronizations;
          each set contains information about topics,
          key --- name of topic, value --- named tuple:
          - *.topic_info[sync_index][topic_name].tokens --- list of top tokens
            for this topic
          - *.topic_info[sync_index][topic_name].weights --- list of weights
            (probabilities), corresponds the tokens
          - *.topic_info[sync_index][topic_name].coherence --- the coherency
            of topic due to it's top tokens
        """
        return self._topic_info

    @property
    def average_coherence(self):
        """Returns:
          list of double: average coherence of top tokens in all requested topics
          on synchronizations
        """
        return self._average_coherence

    @property
    def last_num_tokens(self):
        """Returns:
          int: reqested number of top tokens in each topic on the last
          synchronization
        """
        return self._num_tokens[-1]

    @property
    def last_topic_info(self):
        """Returns:
          set: information about top tokens per topic on the last
          synchronization;
          each set contains information about topics,
          key --- name of topic, value --- named tuple:
          - *.last_topic_info[topic_name].tokens --- list of top tokens
            for this topic
          - *.last_topic_info[topic_name].weights --- list of weights
            (probabilities), corresponds the tokens
          - *.last_topic_info[topic_name].coherence --- the coherency
            of topic due to it's top tokens
        """
        return self._topic_info[-1]

    @property
    def last_average_coherence(self):
        """Returns:
          double: average coherence of top tokens in all requested topics
          on the last synchronization
        """
        return self._average_coherence[-1]


###################################################################################################
class TopicKernelScoreTracker(object):
    """TopicKernelScoreTracker represents a result of counting TopicKernelScore
    (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._topic_info = []
        self._average_coherence = []
        self._average_size = []
        self._average_contrast = []
        self._average_purity = []

    def add(self, score=None):
        """TopicKernelScoreTracker.add() --- add info about score after
        synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.master.retrieve_score(score._model, score.name)

            self._topic_info.append({})
            for topic_index, topic_name in enumerate(_data.topic_name.value):
                tokens = [token for token in _data.kernel_tokens[topic_index].value]
                coherence = -1
                if len(_data.coherence.value) > 0:
                    coherence = _data.coherence.value[topic_index]
                self._topic_info[-1][topic_name] = \
                    collections.namedtuple('TopicKernelScoreTuple',
                                           ['tokens', 'size', 'contrast', 'purity', 'coherence'])
                self._topic_info[-1][topic_name].tokens = tokens
                self._topic_info[-1][topic_name].size = _data.kernel_size.value[topic_index]
                self._topic_info[-1][topic_name].contrast = \
                    _data.kernel_purity.value[topic_index]
                self._topic_info[-1][topic_name].purity = \
                    _data.kernel_contrast.value[topic_index]
                self._topic_info[-1][topic_name].coherence = coherence

            self._average_coherence.append(_data.average_coherence)
            self._average_size.append(_data.average_kernel_size)
            self._average_contrast.append(_data.average_kernel_contrast)
            self._average_purity.append(_data.average_kernel_purity)
        else:
            self._topic_info.append(None)
            self._average_coherence.append(None)
            self._average_size.append(None)
            self._average_contrast.append(None)
            self._average_purity.append(None)

    @property
    def name(self):
        return self._name

    @property
    def topic_info(self):
        """Returns:
          list of sets: information about kernel tokens per topic on
          synchronizations; each set contains information
          about topics, key --- name of topic, value --- named tuple:
          - *.topic_info[sync_index][topic_name].tokens --- list of
            kernel tokens for this topic
          - *.topic_info[sync_index][topic_name].size --- size of
            kernel for this topic
          - *.topic_info[sync_index][topic_name].contrast --- contrast of
            kernel for this topic.
          - *.topic_info[sync_index][topic_name].purity --- purity of kernel
            for this topic
          - *.topic_info[sync_index][topic_name].coherence --- the coherency of
            topic due to it's kernel
        """
        return self._topic_info

    @property
    def average_coherence(self):
        """Returns:
          list of double: average coherence of kernel tokens in all requested
          topics on synchronizations
        """
        return self._average_coherence

    @property
    def average_size(self):
        """Returns:
          list of double: average kernel size of all requested topics on
          synchronizations
        """
        return self._average_size

    @property
    def average_contrast(self):
        """Returns:
          list of double: average kernel contrast of all requested topics on
        synchronizations
        """
        return self._average_contrast

    @property
    def average_purity(self):
        """Returns:
          list of double: average kernel purity of all requested topics on
        synchronizations
        """
        return self._average_purity

    @property
    def last_topic_info(self):
        """Returns:
          set: information about kernel tokens per topic on the last
          synchronization; each set contains information about topics,
          key --- name of topic, value --- named tuple:
          - *.topic_info[topic_name].tokens --- list of
            kernel tokens for this topic
          - *.topic_info[topic_name].size --- size of
            kernel for this topic
          - *.topic_info[topic_name].contrast --- contrast of
            kernel for this topic
          - *.topic_info[topic_name].purity --- purity of kernel
            for this topic
          - *.topic_info[topic_name].coherence --- the coherency of
            topic due to it's kernel
        """
        return self._topic_info[-1]

    @property
    def last_average_coherence(self):
        """Returns:
          double: average coherence of kernel tokens in all requested
          topics on the last synchronization
        """
        return self._average_coherence[-1]

    @property
    def last_average_size(self):
        """Returns:
          double: average kernel size of all requested topics on
          the last synchronization
        """
        return self._average_size[-1]

    @property
    def last_average_contrast(self):
        """Returns:
          double: average kernel contrast of all requested topics on
          the last synchronization
        """
        return self._average_contrast[-1]

    @property
    def last_average_purity(self):
        """Returns:
          double: average kernel purity of all requested topics on
          the last synchronization
        """
        return self._average_purity[-1]


###################################################################################################
class ThetaSnippetScoreTracker(object):
    """ThetaSnippetScoreTracker represents a result of counting
    ThetaSnippetScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._document_ids = []
        self._snippet = []

    def add(self, score=None):
        """ThetaSnippetScoreTracker.add() --- add info about score after
        synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.master.retrieve_score(score._model, score.name)

            self._document_ids.append([item_id for item_id in _data.item_id])
            self._snippet.append(
                [[theta_td for theta_td in theta_d.value] for theta_d in _data.values])
        else:
            self._document_ids.append(None)
            self._snippet.append(None)

    @property
    def name(self):
        return self._name

    @property
    def snippet(self):
        """Returns:
          list of lists of lists of double: the snippet (part) of Theta
          corresponds to documents from document_ids on each synchronizations;
          each most internal list --- theta_d vector for document d,
          in direct order of document_ids
        """
        return self._snippet

    @property
    def document_ids(self):
        """Returns:
          list of int: ids of documents in snippet on synchronizations
        """
        return self._document_ids

    @property
    def last_snippet(self):
        """Returns:
          list of lists of double: the snippet (part) of Theta corresponds
          to documents from document_ids on last synchronization;
          each internal list --- theta_d vector for document d,
          in direct order of document_ids
        """
        return self._snippet

    @property
    def last_document_ids(self):
        """Returns:
          list of int: ids of documents in snippet on the last synchronization
        """
        return self._document_ids


###################################################################################################
class TopicMassPhiScoreTracker(object):
    """TopicMassPhiScoreTracker represents a result of counting
    TopicMassPhiScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._topic_info = []

    def add(self, score=None):
        """TopicMassPhiScoreTracker.add() --- add info about score after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.master.retrieve_score(score._model, score.name)

            self._value.append(_data.value)

            self._topic_info.append({})

            for top_idx, top_name in enumerate(_data.topic_name):
                self._topic_info[-1][top_name] = \
                    collections.namedtuple('TopicMassPhiScoreTuple', ['topic_mass', 'topic_ratio'])
                self._topic_info[-1][top_name].topic_mass = _data.topic_mass[top_idx]
                self._topic_info[-1][top_name].topic_ratio = _data.topic_ratio[top_idx]
        else:
            self._value.append(None)
            self._topic_info.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of double: mass of given topics in Phi on synchronizations
        """
        return self._value

    @property
    def topic_info(self):
        """Returns:
          list of sets: information about topic mass per topic on
          synchronizations; each set contains information
          about topics, key --- name of topic, value --- named tuple:
          - *.topic_info[sync_index][topic_name].topic_mass --- n_t value
          - *.topic_info[sync_index][topic_name].topic_ratio --- p_t value
        """
        return self._topic_info

    @property
    def last_value(self):
        """Returns:
        double: mass of given topics in Phi on last synchronization
        """
        return self._value[-1]

    @property
    def last_topic_info(self):
        """Returns:
          list of sets: information about topic mass per topic on last
          synchronization; each set contains information
          about topics, key --- name of topic, value --- named tuple:
          - *.topic_info[sync_index][topic_name].topic_mass --- n_t value
          - *.topic_info[sync_index][topic_name].topic_ratio --- p_t value
        """
        return self._topic_info[-1]
