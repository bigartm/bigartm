#!/bin/sh

VERSION=$1
BUILDDIR=/tmp/google-glog

# copy sources to clean target
echo -- create $BUILDDIR
mkdir -p $BUILDDIR
echo -- copy sources to $BUILDDIR/$VERSION
cp -rf . $BUILDDIR/$VERSION
CURRENT=`pwd`
cd $BUILDDIR/$VERSION
# remove .svn directories (if any)
echo -- clean svn information
rm -rf `find . -type d -name .svn`

# build src package
echo -- build source package
dpkg-buildpackage -us -uc -S -rfakeroot

# build binary packages
echo -- build binaries

sudo pbuilder build --buildresult $BUILDDIR ../*.dsc 

# copy binaries back to source dir
echo -- retrieve packages
#cd ..
lintian *.deb
mv *.deb ../*.dsc ../*.tar.gz ../*.changes $CURRENT/../

# cleanup
echo -- cleanup
cd $CURRENT
rm -rf $BUILDDIR

echo -- DONE

