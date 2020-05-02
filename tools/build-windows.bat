rem Build wheels (using cibuildwheel) on Windows (test)

echo on

# clone directory
clone_folder: C:\projects\bigartm

set PATH=%MINICONDA%;%MINICONDA%\\Scripts;%PATH%
call conda config --set always_yes yes --set changeps1 no
call conda update -q -c conda-forge conda
call conda info -a
call conda install -c conda-forge numpy scipy pandas pytest
call conda install -c conda-forge tqdm

rem scripts to run before build

md C:\projects\bigartm
cd C:\projects\bigartm
md build
cd build
if "%platform%"=="Win32" set CMAKE_GENERATOR_NAME=Visual Studio 14 2015
if "%platform%"=="x64"   set CMAKE_GENERATOR_NAME=Visual Studio 14 2015 Win64
call cmake -DPYTHON=python -G "%CMAKE_GENERATOR_NAME%" -DCMAKE_BUILD_TYPE=%configuration% -DBOOST_ROOT:PATH="%BOOST_ROOT%" -DBOOST_LIBRARYDIR:PATH="%BOOST_LIBRARYDIR%" ..
rem cd %BIGARTM_UNITTEST_DATA%
rem Start-FileDownload 'https://s3-eu-west-1.amazonaws.com/artm/docword.kos.txt'
rem Start-FileDownload 'https://s3-eu-west-1.amazonaws.com/artm/vocab.kos.txt'

rem build:
rem  project: C:\projects\bigartm\build\INSTALL.vcxproj    # path to Visual Studio solution or project
rem  parallel: false                                       # enable MSBuild parallel builds
rem  verbosity: normal                                     # MSBuild verbosity level (quiet|minimal|normal|detailed)

rem after_build:
rem - cmd: cd C:\projects\bigartm\3rdparty\protobuf-3.0.0\python
rem - cmd: python setup.py install
rem - cmd: cd C:\projects\bigartm\python
rem - cmd: python setup.py install
rem - cmd: call C:\projects\bigartm\utils\create_windows_package.bat

rem test_script:
rem - cmd: cd C:\projects\bigartm\build\bin\Release
rem - cmd: artm_tests.exe
rem - cmd: cd C:\projects\bigartm\python\tests
rem - cmd: py.test
