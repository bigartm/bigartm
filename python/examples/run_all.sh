#!/bin/sh

for f in *.py
do
    echo "==== $f ===="
    python "$f" || exit 1
done
