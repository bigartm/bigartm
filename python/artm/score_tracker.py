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


def _set_properties(class_ref, attr_data):
    for name, params in attr_data.iteritems():
        _p = [name, 'optional', 'scalar', 'topic_name']
        for k, v in params.iteritems():
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


SparsityPhiScoreTracker = type('SparsityPhiScoreTracker', (BaseScoreTracker, ), {})
_set_properties(SparsityPhiScoreTracker, {'value': {}, 'zero_tokens': {}, 'total_tokens': {}})


SparsityThetaScoreTracker = type('SparsityThetaScoreTracker', (BaseScoreTracker, ), {})
_set_properties(SparsityThetaScoreTracker, {'value': {}, 'zero_topics': {}, 'total_topics': {}})


PerplexityScoreTracker = type('PerplexityScoreTracker', (BaseScoreTracker, ), {})
_set_properties(PerplexityScoreTracker, {'value': {}, 'raw': {}, 'normalizer': {},
                                         'zero_tokens': {'proto_name': 'zero_words'},
                                         'theta_sparsity_value': {},
                                         'theta_sparsity_zero_topics': {},
                                         'theta_sparsity_total_topics': {}})


ItemsProcessedScoreTracker = type('ItemsProcessedScoreTracker', (BaseScoreTracker, ), {})
_set_properties(ItemsProcessedScoreTracker, {'value': {}})


TopTokensScoreTracker = type('TopTokensScoreTracker', (BaseScoreTracker, ), {})
_set_properties(TopTokensScoreTracker, {'num_tokens': {'proto_name': 'num_entries'},
                                        'tokens': {'proto_name': 'token',
                                                   'proto_qualifier': 'repeated'},
                                        'weights': {'proto_name': 'weight',
                                                    'proto_qualifier': 'repeated'},
                                        'coherence': {'proto_type': 'array'},
                                        'average_coherence': {}})


TopicKernelScoreTracker = type('TopicKernelScoreTracker', (BaseScoreTracker, ), {})
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


ThetaSnippetScoreTracker = type('ThetaSnippetScoreTracker', (BaseScoreTracker, ), {})
_set_properties(ThetaSnippetScoreTracker, {'snippet': {'proto_name': 'values',
                                                       'proto_qualifier': 'repeated',
                                                       'proto_type': 'array',
                                                       'key_field_name': 'item_id'},
                                           'document_ids': {'proto_name': 'item_id',
                                                            'proto_qualifier': 'repeated',
                                                            'key_field_name': None}})


TopicMassPhiScoreTracker = type('TopicMassPhiScoreTracker', (BaseScoreTracker, ), {})
_set_properties(TopicMassPhiScoreTracker, {'value': {},
                                           'topic_mass': {'proto_qualifier': 'repeated'},
                                           'topic_ratio': {'proto_qualifier': 'repeated'}})


ClassPrecisionScoreTracker = type('ClassPrecisionScoreTracker', (BaseScoreTracker, ), {})
_set_properties(ClassPrecisionScoreTracker, {'value': {}, 'error': {}, 'total': {}})
