#!/bin/bash
#

# cibuildwheel builder for MacOS

# -e is "exits as soon as any line in the bash script fails"
# set -ex

# -x is "prints each command that is going to be executed with a little plus"
set -x


echo "# INITIALIZING ENV VARS"
if [ -z $TRAVIS_BUILD_DIR ]; then
    echo "# TRAVIS_BUILD_DIR is empty"
    export CI_BUILD_DIR='/project'
else
    export CI_BUILD_DIR=$TRAVIS_BUILD_DIR
fi

echo $CI_BUILD_DIR


# on Travis, boost 1.71.0 is already installed

pip install -U pip -q
pip install -U pytest pep8 wheel==0.34.1 protobuf==3.0.0 numpy scipy pandas tqdm --only-binary numpy scipy pandas -q

cd $CI_BUILD_DIR

