import os
import sys

def find_bigartm_binary():
    binary_name = os.environ.get('BIGARTM_EXECUTABLE')

    if binary_name is None:
        if sys.platform.startswith('win'):
            binary_name = 'bigartm.exe'
        else:
            binary_name = 'bigartm'

    return binary_name
