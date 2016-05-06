set BIGARTM_VERSION="v0.8.0"

call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\vcvarsall.bat"
set VSVERSION=11

set CONFIG=debug
set FLOVER=32
call build.bat

set CONFIG=RelWithDebInfo
set FLOVER=32
call build.bat

set CONFIG=debug
set FLOVER=64
call build.bat

set CONFIG=RelWithDebInfo
set FLOVER=64
call build.bat

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"
set VSVERSION=12

set CONFIG=debug
set FLOVER=32
call build.bat

set CONFIG=RelWithDebInfo
set FLOVER=32
call build.bat

set CONFIG=debug
set FLOVER=64
call build.bat

set CONFIG=RelWithDebInfo
set FLOVER=64
call build.bat
