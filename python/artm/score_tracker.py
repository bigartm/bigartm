# Copyright 2017, Additive Regularization of Topic Models.

from six import iteritems
from six.moves import zip


__all__ = [
    'PerplexityScoreTracker',
    'SparsityThetaScoreTracker',
    'SparsityPhiScoreTracker',
    'ItemsProcessedScoreTracker',
    'TopTokensScoreTracker',
    'ThetaSnippetScoreTracker',
    'TopicKernelScoreTracker',
    'TopicMassPhiScoreTracker',
    'ClassPrecisionScoreTracker',
    'BackgroundTokensRatioScoreTracker'
]


def _get_score(score_name, master, field_attrs, last=False):
    def __getattr(data, field):
        try:
            _ = (e for e in getattr(data, field))
        except TypeError:
            return getattr(data, field).value

        return getattr(data, field)

    def __create_dict(keys, values):
        result_dict = {}
        for k, v in zip(keys, values):
            if k not in result_dict:
                result_dict[k] = []
            result_dict[k].append(v)

        if len(keys) == len(result_dict.keys()):
            for k in result_dict.keys():
                result_dict[k] = result_dict[k][-1]

        return result_dict

    data_array = master.get_score_array(score_name)

    if field_attrs[1] == 'optional' and field_attrs[2] == 'scalar':
        score_list = [getattr(data, field_attrs[0]) for data in data_array]
        return score_list[-1] if last else score_list

    else:
        score_list_list = [__getattr(data, field_attrs[0]) for data in data_array]

        if ((field_attrs[1] == 'repeated' and field_attrs[2] == 'scalar') or
            (field_attrs[1] == 'optional' and field_attrs[2] == 'array')):  # noqa
            if field_attrs[3] is None:
                return score_list_list[-1] if last else score_list_list
            else:
                score_topic_list_list = (zip(score_list_list,
                    [__getattr(data, field_attrs[3]) for data in data_array]))  # noqa
                score_list_dict = [__create_dict(t, s) for s, t in score_topic_list_list]
                return score_list_dict[-1] if last else score_list_dict

        elif field_attrs[1] == 'repeated' and field_attrs[2] == 'array':
            score_topic_list_list = zip(score_list_list,
                                        [__getattr(data, field_attrs[3]) for data in data_array])
            score_list_dict = ([{topic: score_array.value for (score_array, topic) in
                zip(score_arrays, topics)} for score_arrays, topics in score_topic_list_list])  # noqa
            return score_list_dict[-1] if last else score_list_dict

        elif field_attrs[1] == 'repeated' and field_attrs[2] == 'struct':
            score_list_dict = [{__getattr(s, field_attrs[3]): s for s in score_list}
                               for score_list in score_list_list]  # noqa
            return score_list_dict[-1] if last else score_list_dict
        else:
            raise ValueError('Unkown type of score tracker field')


def _set_properties(class_ref, attr_data):
    for name, params in iteritems(attr_data):
        _p = [name, 'optional', 'scalar', 'topic_name']
        for k, v in iteritems(params):
            _p[0] = v if k == 'proto_name' else _p[0]
            _p[1] = v if k == 'proto_qualifier' else _p[1]
            _p[2] = v if k == 'proto_type' else _p[2]
            _p[3] = v if k == 'key_field_name' else _p[3]

        setattr(class_ref,
                name,
                property(lambda self, p=_p: _get_score(self._name, self._master, p)))
        setattr(class_ref,
                'last_{}'.format(name),
                property(lambda self, p=_p: _get_score(self._name, self._master, p, True)))


class BaseScoreTracker(object):
    def __init__(self, score):
        self._name = score.name
        self._master = score.master


class SparsityPhiScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * value - values of Phi sparsity.
        * zero_tokens - number of zero rows in Phi.
        * total_tokens - number of all rows in Phi.
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(SparsityPhiScoreTracker, {'value': {}, 'zero_tokens': {}, 'total_tokens': {}})


class SparsityThetaScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * value - values of Theta sparsity.
        * zero_topics - number of zero rows in Theta.
        * total_topics - number of all rows in Theta.
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(SparsityThetaScoreTracker, {'value': {}, 'zero_topics': {}, 'total_topics': {}})


class PerplexityScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * value - values of perplexity.
        * raw - raw values in formula for perplexity (in case of one class id).
        * normalizer - normalizer values in formula for perplexity  (in case of one class id).
        * zero_tokens - number of zero p(w|d) = sum_t p(w|t) p(t|d)  (in case of one class id).
        * class_id_info - array of structures, each structure contains raw, normalizer\
                          zero_tokens and class_id name (in case of several class ids).
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(PerplexityScoreTracker, {'value': {}, 'raw': {}, 'normalizer': {},
                                         'zero_tokens': {'proto_name': 'zero_words'},
                                         'class_id_info': {'proto_qualifier': 'repeated',
                                                           'proto_type': 'struct',
                                                           'key_field_name': 'class_id'}})


class ItemsProcessedScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * value - numbers of processed documents.
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(ItemsProcessedScoreTracker, {'value': {}})


class TopTokensScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * num_tokens - number of requested top tokens.
        * coherence - each element of list is a dict, key - topic name, value - topic coherence\
                      counted using top-tokens
        * average_coherence - average coherencies of all scored topics.
        * tokens - each element of list is a dict, key - topic name, value - list of top-tokens
        * weights - each element of list is a dict, key - topic name, value - list of weights of\
                    corresponding top-tokens (weight of token == p(w|t))
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(TopTokensScoreTracker, {'num_tokens': {'proto_name': 'num_entries'},
                                        'tokens': {'proto_name': 'token',
                                                   'proto_qualifier': 'repeated'},
                                        'weights': {'proto_name': 'weight',
                                                    'proto_qualifier': 'repeated'},
                                        'coherence': {'proto_type': 'array'},
                                        'average_coherence': {}})


class TopicKernelScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * tokens - each element of list is a dict, key - topic name, value - list of kernel tokens
        * size - each element of list is a dict, key - topic name, value - kernel size
        * contrast - each element of list is a dict, key - topic name, value - kernel contrast
        * purity - each element of list is a dict, key - topic name, value - kernel purity
        * coherence - each element of list is a dict, key - topic name, value - topic coherence\
                      counted using kernel tokens
        * average_size - average kernel size of all scored topics.
        * average_contrast - average kernel contrast of all scored topics.
        * average_purity - average kernel purity of all scored topics.
        * average_coherence - average coherencies of all scored topics.
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(TopicKernelScoreTracker, {'tokens': {'proto_name': 'kernel_tokens',
                                                     'proto_qualifier': 'repeated',
                                                     'proto_type': 'array'},
                                          'size': {'proto_name': 'kernel_size',
                                                   'proto_type': 'array'},
                                          'contrast': {'proto_name': 'kernel_contrast',
                                                       'proto_type': 'array'},
                                          'purity': {'proto_name': 'kernel_purity',
                                                     'proto_type': 'array'},
                                          'coherence': {'proto_type': 'array'},
                                          'average_size': {'proto_name': 'average_kernel_size'},
                                          'average_contrast': {'proto_name': 'average_kernel_contrast'},
                                          'average_purity': {'proto_name': 'average_kernel_purity'},
                                          'average_coherence': {}})


class ThetaSnippetScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * document_ids - each element of list is a list of ids of returned documents.
        * snippet - each element of list is a dict, key - doc id, value - list with\
                    corresponding p(t|d) values.
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(ThetaSnippetScoreTracker, {'snippet': {'proto_name': 'values',
                                                       'proto_qualifier': 'repeated',
                                                       'proto_type': 'array',
                                                       'key_field_name': 'item_id'},
                                           'document_ids': {'proto_name': 'item_id',
                                                            'proto_qualifier': 'repeated',
                                                            'key_field_name': None}})


class TopicMassPhiScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * value - values of ratio of sum_t n_t of scored topics.and all topics
        * topic_mass - each value is a dict, key - topic name, value - topic mass n_t
        * topic_ratio - each value is a dict, key - topic name, value - topic ratio
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(TopicMassPhiScoreTracker, {'value': {},
                                           'topic_mass': {'proto_qualifier': 'repeated'},
                                           'topic_ratio': {'proto_qualifier': 'repeated'}})


class ClassPrecisionScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * value - values of ratio of correct predictions.
        * error - numbers of error predictiona.
        * total - numbers of all predictions.
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(ClassPrecisionScoreTracker, {'value': {}, 'error': {}, 'total': {}})


class BackgroundTokensRatioScoreTracker(BaseScoreTracker):
    def __init__(self, score):
        """
        :Properties:
        * Note: every field is a list of info about score on all synchronizations.
        * value - values of part of background tokens.
        * tokens - each element of list is a lists of background tokens\
                   (can be acceced if 'save_tokens' was True)
        * Note: every field has a version with prefix 'last_', means retrieving only\
          info about the last synchronization.
        """
        BaseScoreTracker.__init__(self, score)

_set_properties(BackgroundTokensRatioScoreTracker, {'value': {}, 'tokens': {'proto_name': 'token'}})
