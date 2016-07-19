#!/bin/bash

wget https://s3-eu-west-1.amazonaws.com/artm/docword.kos.txt.gz
gzip -d docword.kos.txt.gz

wget https://s3-eu-west-1.amazonaws.com/artm/vocab.kos.txt
