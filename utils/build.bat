REM =========================================================================================
REM This script is a convenience helper used to build multiple BigARTM versions.
REM It is not recommended for BigARTM users to use this script.
REM Refer to http://bigartm.org for detailed instructions on how to build BigARTM on Windows.
REM =========================================================================================

REM Set this variables:
REM set CONFIG=[debug|RelWithDebInfo]
REM set FLOVER=[32|64]
REM set VSVERSION=[11|12]
REM set BIGARTM_VERSION=...

if "%FLOVER%" == "32" (
  set INSTALL_FOLDER="C:\Program Files (x86)\BigARTM"
  set CMAKE_OPTIONS="Visual Studio %VSVERSION%"
) else (
  set INSTALL_FOLDER="C:\Program Files\BigARTM"
  set CMAKE_OPTIONS="Visual Studio %VSVERSION% Win64"
)
set BOOST_ROOT=C:\local\boost_1_57_0
set BOOST_LIBRARYDIR=C:\local\boost_1_57_0\lib%FLOVER%-msvc-%VSVERSION%.0

set TARGET_PACKAGE="BigARTM_%BIGARTM_VERSION%_vs%VSVERSION%_win%FLOVER%_%CONFIG%.7z"

REM  CD to root folder
cd C:\Users\Administrator\Documents\GitHub\bigartm

REM  Cleanup
rmdir build /s /q
rmdir %INSTALL_FOLDER% /s /q
if EXIST %TARGET_PACKAGE% (del %TARGET_PACKAGE%)

REM  Build
mkdir build && cd build
cmake .. -G%CMAKE_OPTIONS%
if %errorlevel% neq 0 exit /b %errorlevel%

msbuild /p:Configuration=%CONFIG% INSTALL.vcxproj
if %errorlevel% neq 0 exit /b %errorlevel%

REM Copy libs

mkdir %INSTALL_FOLDER%\lib
cp lib\%CONFIG%\artm.lib %INSTALL_FOLDER%\lib\artm.lib
cp lib\%CONFIG%\protobuf.lib %INSTALL_FOLDER%\lib\protobuf.lib

REM  Create the package
cd ..
"C:\Program Files\7-Zip\7z.exe" a %TARGET_PACKAGE% %INSTALL_FOLDER%\*

