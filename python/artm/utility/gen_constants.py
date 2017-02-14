# Copyright 2017, Additive Regularization of Topic Models.

import os
from datetime import date

names_stack = []
constants = []
is_enum = False

with open(os.path.join('..', '..', '..', 'src', 'artm', 'messages.proto'), 'r') as fin:
    for line in fin:
        if '}' in line:
            if len(names_stack) > 0:
                names_stack.pop()
            is_enum = False
            continue

        line_list = line.strip().split(' ')
        if len(line_list) < 2:
            continue

        if is_enum:
            const_name, number = line_list[0], line_list[2]
            current_str = ''
            for name in names_stack[: -1] if const_name.startswith(names_stack[-1]) else names_stack:
                current_str += '{}_'.format(name)
            current_str += '{0} = {1}'.format(const_name, number[: -1])
            constants.append(current_str)
            continue

        first_token, mes_name = line_list[0], line_list[1]
        is_enum = (first_token == 'enum')
        if is_enum or first_token == 'message':
            names_stack.append(mes_name)
            continue

with open(os.path.join('..', 'wrapper', 'constants.py'), 'w') as fout:
    fout.write('# Copyright {}, Additive Regularization of Topic Models.\n'.format(date.today().year))
    fout.write('\n"""\n')
    fout.write('Constants values, used in messages\n')
    fout.write('This file was generated using python/artm/utility/gen_constants.py\n')
    fout.write('Don\'t modify this file by hand!\n')
    fout.write('"""\n\n')

    for line in constants:
        fout.write('{}\n'.format(line))
