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

# hack needed to install boost only one
# see: https://github.com/joerick/cibuildwheel/issues/54
if [ ! -f built-lib ]; then
    echo "# Installing basic system dependencies"
    # brew install -y bzip2-devel zip
    curl -L http://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz -o boost_1_60_0.tar.gz && tar -xf boost_1_60_0.tar.gz && cd boost_1_60_0 && ./bootstrap.sh 

    # we are in an awkward state of "log is too big for travis to handle" and "no output for 20 minutes, travis declares us dead" 
    # travis_wait does not work inside CentOS docker (why should it?)
    # see also: https://github.com/CCPPETMR/SIRF-SuperBuild/issues/177
    ./b2 link=static,shared cxxflags="-std=c++11 -fPIC" --without-python
    # ./b2 install --without-python -d0 --prefix=$CI_BUILD_DIR

    sudo ./b2 install --without-python -d0

    touch built-lib
fi

pip install -U pip -q
pip install -U pytest pep8 wheel==0.31.1 protobuf==3.0.0 numpy scipy pandas tqdm --only-binary numpy scipy pandas -q

cd $CI_BUILD_DIR
pwd
ls

if [ -d $CI_BUILD_DIR/build ]; then rm -rf build; fi
mkdir $CI_BUILD_DIR/build && cd $CI_BUILD_DIR/build

# cmake -DPYTHON="${PYBIN}/python" -DBUILD_TESTS=OFF -DBoost_USE_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr ..

cmake -DPYTHON=python -DBUILD_TESTS=OFF -DBoost_USE_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr ..

# dirty hack to fix librt issue
cat src/bigartm/CMakeFiles/bigartm.dir/link.txt | awk '{print $0 " -lrt"}' > src/bigartm/CMakeFiles/bigartm.dir/link2.txt && mv -f src/bigartm/CMakeFiles/bigartm.dir/link2.txt src/bigartm/CMakeFiles/bigartm.dir/link.txt

make
 
pwd
ls


