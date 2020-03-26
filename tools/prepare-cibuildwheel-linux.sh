#!/bin/bash
#

# cibuildwheel builder for Linux
# based on: https://github.com/vengky/confluent-kafka-python/blob/master/tools/prepare-cibuildwheel-linux.sh

set -ex

echo "# Installing basic system dependencies"
yum install -y bzip2-devel zip
curl -L http://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz -o boost_1_60_0.tar.gz && tar -xf boost_1_60_0.tar.gz && cd boost_1_60_0 && ./bootstrap.sh && ./b2 link=static,shared cxxflags="-std=c++11 -fPIC" --without-python && ./b2 install --without-python



