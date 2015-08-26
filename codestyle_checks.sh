#!/bin/bash

echo "Running C++ code checks"
cat utils/cpplint_files.txt | xargs python utils/cpplint.py --linelength=120 || exit 1

echo "Running Python code checks"
for scr in python/artm/{model,batches_utils,regularizers,scores,score_tracker,master_component}.py
do
    pep8 --first --max-line-length=100 ${scr} || exit 1
done
