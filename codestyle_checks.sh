#!/bin/bash

echo "Running C++ code checks"
cat utils/cpplint_files.txt | xargs python utils/cpplint.py --linelength=120 || exit 1

echo "Running Python code checks"
pep8 --first --max-line-length=99 python/artm/model.py || exit 1
pep8 --first --max-line-length=99 python/artm/batches.py || exit 1
pep8 --first --max-line-length=99 python/artm/regularizers.py || exit 1
pep8 --first --max-line-length=99 python/artm/scores.py || exit 1
