#!/bin/bash
#
#
# based on https://github.com/vengky/confluent-kafka-python/blob/master/tools/build-manylinux.sh
# and some magic
# 
# Builds autonomous Python packages including all dependencies
# using the excellent manylinux docker images and the equally awesome
# auditwheel tool.
#
# This script should be run in a docker image where the confluent-kafka-python
# directory is mapped as /io .
#
# Usage on host:
#  tools/build-manylinux.sh <librdkafka_tag>
#
# Usage in container:
#  docker run -t -v $(pwd):/io quay.io/pypa/manylinux1_x86_64:latest  /io/tools/build-manylinux.sh <librdkafka_tag>

set -ex

if [[ ! -f /.dockerenv ]]; then
    #
    # Running on host, fire up a docker container and run it.
    #

    if [[ ! -f tools/$(basename $0) ]]; then
        echo "Must be called from (?) root directory"
        exit 1
    fi

    docker run -t -v $(pwd):/io quay.io/pypa/manylinux1_x86_64:latest  /io/tools/build-manylinux.sh

    exit $?
fi


#
# Running in container
#

echo "# Installing basic system dependencies"
yum install -y bzip2-devel zip
curl -L http://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz -o boost_1_60_0.tar.gz && tar -xf boost_1_60_0.tar.gz && cd boost_1_60_0 && ./bootstrap.sh && ./b2 link=static,shared cxxflags="-std=c++11 -fPIC" --without-python && ./b2 install --without-python

# manylinux image came with pre-installed cmake 2.8.11.2, while protobuf-3 requires cmake 2.8.12.
# So, we have to manually install a newer version of cmake.
# Instructions taken from https://askubuntu.com/questions/355565/how-to-install-latest-cmake-version-in-linux-ubuntu-from-command-line.
mkdir ~/temp && cd ~/temp && curl -L https://cmake.org/files/v3.9/cmake-3.9.1.tar.gz -o cmake-3.9.1.tar.gz && tar -xzvf cmake-3.9.1.tar.gz && cd cmake-3.9.1/ && ./bootstrap && make && make install && cd ~ && rm -rf ~/temp && cmake --version

for PYBIN in /opt/python/*/bin; do\
    "${PYBIN}/pip" install -U pip 
    "${PYBIN}/pip" install -U pytest pep8 wheel==0.31.1 protobuf==3.0.0 numpy scipy pandas tqdm --only-binary numpy scipy pandas;\
done

mkdir /bigartm9/ && cd /bigartm9/

git clone --branch v0.9.x --depth=1 https://github.com/bigartm/bigartm.git

mkdir /bigartm10/ && cd /bigartm10/
git clone --branch master --depth=1 https://github.com/bigartm/bigartm.git

cd /

mkdir /wheelhouse/
for ARTMVER in bigartm9 bigartm10; do

        cd /$ARTMVER/bigartm
        echo -e "[install]\ninstall_lib=" >> python/setup.cfg

        for PYBIN in /opt/python/*/bin; do
            echo "## Compiling $PYBIN"
            mkdir build && cd build
            cmake -DPYTHON="${PYBIN}/python" -DBUILD_TESTS=OFF -DBoost_USE_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr ..
            cat ./src/bigartm/CMakeFiles/bigartm.dir/link.txt | awk '{print $0 " -lrt"}' > ./src/bigartm/CMakeFiles/bigartm.dir/link2.txt && mv -f ./src/bigartm/CMakeFiles/bigartm.dir/link2.txt ./src/bigartm/CMakeFiles/bigartm.dir/link.txt
            make
            auditwheel repair ./python/bigartm*-linux_x86_64.whl
            cp ./wheelhouse/bigartm*-manylinux1_x86_64.whl /wheelhouse/
            cd .. && rm -rf build
        done
done

echo "# Repaired wheels"
for whl in /wheelhouse/*.whl; do
    echo "## Repaired wheel $whl"
    auditwheel show "$whl"
done

zip -r /wheels /wheelhouse/*


# Install packages and test
# echo "# Installing wheels"
# for PYBIN in /opt/python/*/bin/; do
#    # for ARTMVER in bigartm9 bigartm10; do
#        # echo "## Installing $ARTMVER on $PYBIN"
#        # "${PYBIN}/pip" install $ARTMVER -f /wheelhouse
#        # "${PYBIN}/python" -c 'import artm; print(artm.version())'
#        # echo "## Uninstalling $PYBIN"
#        # "${PYBIN}/pip" uninstall -y bigartm
# done




