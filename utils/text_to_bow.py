# Convert plaintext into UCI Bag-of-words
#
# Created by Albert Aparicio <aaparicio@posteo.net>

from __future__ import print_function

import argparse
import os

from sklearn.feature_extraction.text import CountVectorizer


def main(args):
  file_dir = os.path.dirname(args.fname)

  basename, ext = os.path.splitext(os.path.basename(args.fname))

  # Read text file
  with open(args.fname, 'r')as file:
    rawdata = file.read().replace('\n', '')

  # instantiate the parser and feed it some HTML
  vectorizer = CountVectorizer()

  x = vectorizer.fit_transform([rawdata])

  vocabulary = vectorizer.get_feature_names()
  count_vector = x.toarray()

  # Initialize Bag-of-words data
  bow = [
    str(1),  # D - Number of documents
    str(len(vocabulary)),  # W - Number of words in vocabulary
    str(count_vector.sum()),  # NNZ - Number of words in documents
    ]

  for word, count in zip(vocabulary, count_vector.squeeze()):
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
