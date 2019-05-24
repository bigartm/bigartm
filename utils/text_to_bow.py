# Convert plaintext into UCI Bag-of-words
#
# Created by Albert Aparicio <aaparicio@posteo.net>

from __future__ import print_function

import argparse
import os
import re
from collections import Counter

from six import iteritems


def main(args):
  file_dir = os.path.dirname(args.fname)

  basename, ext = os.path.splitext(os.path.basename(args.fname))

  # Read text file
  with open(args.fname, 'r')as file:
    rawdata = file.read().replace('\n', '')

  bagofwords = Counter(re.findall(r'\w+', rawdata.lower()))
  vocabulary = sorted(bagofwords.keys())

  # Initialize Bag-of-words data
  bow = [
    str(1),  # D - Number of documents
    str(len(bagofwords)),  # W - Number of words in vocabulary
    str(sum(bagofwords.values())),  # NNZ - Number of words in documents
    ]

  for word, count in iteritems(bagofwords):
    bow.append('{} {} {}'.format(1, 1 + vocabulary.index(word), count))

  # Save docfile and vocabulary
  with open(os.path.join(file_dir, 'docword.{}.txt'.format(basename)), 'w') as docf:
    docf.writelines('\n'.join(bow))

  with open(os.path.join(file_dir, 'vocab.{}.txt'.format(basename)), 'w') as vocabf:
    vocabf.writelines('\n'.join(vocabulary))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Convert plaintext to UCI Bag-of-words')
  parser.add_argument('-f', '--fname', type=str, required=True,
                      help='Filename of the data to convert')

  opts = parser.parse_args()

  print('Arguments parsed')

  main(opts)

  exit()
