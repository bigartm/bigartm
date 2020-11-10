#!/bin/bash
#

# cibuildwheel builder for MacOS

# -e is "exits as soon as any line in the bash script fails"
# set -ex

# -x is "prints each command that is going to be executed with a little plus"
set -x



# on Travis, boost 1.71.0 is already installed
# but it seems that we need to control linking differently?

# hack needed to install boost only once
# see: https://github.com/joerick/cibuildwheel/issues/54
# if [ ! -f built-lib ]; then
if false; then
    echo "# Installing basic system dependencies"
    curl -L http://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz -o boost_1_60_0.tar.gz && tar -xf boost_1_60_0.tar.gz && cd boost_1_60_0 && ./bootstrap.sh 

    # we are in an awkward state of "log is too big for travis to handle" and "no output for 20 minutes, travis declares us dead" 
    # TODO: use -d0 and travis_wait here
    ./b2 runtime-link=shared link=static,shared cxxflags="-std=c++11 -fPIC" --without-python

    ./b2 install --without-python -d0

    ./bootstrap > /dev/null
    make -s
    make install -s
    cd ~

    touch built-lib
else
    echo "# boost is already installed, skipping ..."
fi



pip install -U pip -q
pip install -U pytest pep8 protobuf==3.0.0 numpy scipy pandas tqdm --only-binary numpy scipy pandas -q


