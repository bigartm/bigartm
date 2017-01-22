# Copyright 2017, Additive Regularization of Topic Models.

"""
Auxiliary functions used in wrapper
"""
from six import iteritems


def dict_to_message(record, message_type):
    """Convert dict to protobuf message"""

    def parse_list(values, message):
        if isinstance(values[0], dict):
            for v in values:
                cmd = message.add()
                parse_dict(v, cmd)
        else:
            message.extend(values)

    def parse_dict(values, message):
        for k, v in iteritems(values):
            if isinstance(v, dict):
                parse_dict(v, getattr(message, k))
            elif isinstance(v, list):
                parse_list(v, getattr(message, k))
            else:
                try:
                    setattr(message, k, v)
                except AttributeError:
                    raise TypeError('Cannot convert dict to protobuf message '
                                    '{message_type}: bad field "{field}"'.format(
                            message_type=str(message_type),
                            field=k,
                        ))
    
    message = message_type()
    parse_dict(record, message)

    return message
