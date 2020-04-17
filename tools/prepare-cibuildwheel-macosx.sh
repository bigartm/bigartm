#!/bin/bash
#

# cibuildwheel builder for MacOS

# -e is "exits as soon as any line in the bash script fails"
# set -ex

# -x is "prints each command that is going to be executed with a little plus"
set -x



# on Travis, boost 1.71.0 is already installed

pip install -U pip -q
pip install -U pytest pep8 protobuf==3.0.0 numpy scipy pandas tqdm --only-binary numpy scipy pandas -q


