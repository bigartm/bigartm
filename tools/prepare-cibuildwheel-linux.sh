#!/bin/bash
#

# cibuildwheel builder for Linux
# based on: https://github.com/vengky/confluent-kafka-python/blob/master/tools/prepare-cibuildwheel-linux.sh

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

# hack needed to install boost only one
# see: https://github.com/joerick/cibuildwheel/issues/54
if [ ! -f built-lib ]; then
    echo "# Installing basic system dependencies"
    yum install -y bzip2-devel zip
    curl -L http://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz -o boost_1_60_0.tar.gz && tar -xf boost_1_60_0.tar.gz && cd boost_1_60_0 && ./bootstrap.sh 

    # we are in an awkward state of "log is too big for travis to handle" and "no output for 20 minutes, travis declares us dead" 
    # travis_wait does not work inside CentOS docker (why should it?)
    # see also: https://github.com/CCPPETMR/SIRF-SuperBuild/issues/177
    echo $AUIDITWHEEL_PLAT
    if [[ $AUIDITWHEEL_PLAT == manylinux1_x86_64 ]]; then
        ./b2 runtime-link=shared link=static,shared cxxflags="-std=c++11 -fPIC" --without-python -d0
    else
        ./b2 runtime-link=shared link=static,shared cxxflags="-std=c++11 -fPIC" --without-python
    fi

    ./b2 install --without-python -d0

    # manylinux image came with pre-installed cmake 2.8.11.2, while protobuf-3 requires cmake 2.8.12.
    # So, we have to manually install a newer version of cmake.
    # Instructions taken from https://askubuntu.com/questions/355565/how-to-install-latest-cmake-version-in-linux-ubuntu-from-command-line.

    if [ -d ~/temp_cmake ]; then rm -rf ~/temp_cmake; fi
    mkdir ~/temp_cmake
    cd ~/temp_cmake
    curl -L https://cmake.org/files/v3.9/cmake-3.9.1.tar.gz -o cmake-3.9.1.tar.gz && tar -xzf cmake-3.9.1.tar.gz && cd cmake-3.9.1/ 

    ./bootstrap > /dev/null
    make -s
    make install -s
    cd ~ && rm -rf ~/temp_cmake

    touch built-lib
fi

pip install -U pip -q
pip install -U pytest pep8 wheel==0.34.1 protobuf==3.0.0 numpy scipy pandas tqdm --only-binary numpy scipy pandas -q

cd $CI_BUILD_DIR

