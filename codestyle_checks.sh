#!/bin/bash

echo "Running C++ code checks"
cat utils/cpplint_files.txt | xargs python utils/cpplint.py --linelength=120 || exit 1

echo "Running Python code checks"
for scr in python/artm/{model,batches,regularizers,scores}.py
do
    pep8 --first --max-line-length=99 ${scr} || exit 1
done
