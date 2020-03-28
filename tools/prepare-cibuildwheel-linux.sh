#!/bin/bash
#

# cibuildwheel builder for Linux
# based on: https://github.com/vengky/confluent-kafka-python/blob/master/tools/prepare-cibuildwheel-linux.sh

# -e is "exits as soon as any line in the bash script fails"
# set -ex

# -x is "prints each command that is going to be executed with a little plus"
set -x

# hack needed to install boost only one
# see: https://github.com/joerick/cibuildwheel/issues/54
if [ ! -f built-lib ]; then
    echo "# Installing basic system dependencies"
    yum install -y bzip2-devel zip
    curl -L http://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz -o boost_1_60_0.tar.gz && tar -xf boost_1_60_0.tar.gz && cd boost_1_60_0 && ./bootstrap.sh 

    ./b2 link=static,shared cxxflags="-std=c++11 -fPIC" --without-python
    ./b2 install --without-python -d0

    # manylinux image came with pre-installed cmake 2.8.11.2, while protobuf-3 requires cmake 2.8.12.
    # So, we have to manually install a newer version of cmake.
    # Instructions taken from https://askubuntu.com/questions/355565/how-to-install-latest-cmake-version-in-linux-ubuntu-from-command-line.

    if [ -d ~/temp_cmake ]; then rm -rf ~/temp_cmake; fi
    mkdir ~/temp_cmake
    cd ~/temp_cmake
    curl -L https://cmake.org/files/v3.9/cmake-3.9.1.tar.gz -o cmake-3.9.1.tar.gz && tar -xzf cmake-3.9.1.tar.gz && cd cmake-3.9.1/ 

    ./bootstrap > /dev/null
    make
    make install
    cd ~ && rm -rf ~/temp_cmake

    touch built-lib
fi

pip install -U pip -q
pip install -U pytest pep8 wheel==0.31.1 protobuf==3.0.0 numpy scipy pandas tqdm --only-binary numpy scipy pandas -q

cd $TRAVIS_BUILD_DIR
pwd
ls

if [ -d $TRAVIS_BUILD_DIR/build ]; then rm -rf build; fi
mkdir $TRAVIS_BUILD_DIR/build && cd $TRAVIS_BUILD_DIR/build

# cmake -DPYTHON="${PYBIN}/python" -DBUILD_TESTS=OFF -DBoost_USE_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr ..

cmake -DPYTHON=python -DBUILD_TESTS=OFF -DBoost_USE_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr ..

# dirty hack to fix librt issue
cat src/bigartm/CMakeFiles/bigartm.dir/link.txt | awk '{print $0 " -lrt"}' > src/bigartm/CMakeFiles/bigartm.dir/link2.txt && mv -f src/bigartm/CMakeFiles/bigartm.dir/link2.txt src/bigartm/CMakeFiles/bigartm.dir/link.txt

make
 
pwd
ls


