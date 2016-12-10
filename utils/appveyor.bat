REM This file can be executed on appveyor build agent to start IDE with all environment variables set correctly (e.g. as druring build phase)
set INSTALL_FOLDER="C:\Program Files\BigARTM"
set PROTOC=C:\projects\bigartm\build\bin\Release\protoc.exe
set MINICONDA=C:\Miniconda-x64
set PATH=%MINICONDA%;%MINICONDA%\\Scripts;%PATH%
set PROTOC=C:\projects\bigartm\build\bin\Release\protoc.exe
set ARTM_SHARED_LIBRARY=C:\projects\bigartm\build\bin\Release\artm.dll
set BIGARTM_UNITTEST_DATA=C:\projects\bigartm\test_data
"c:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe" C:\projects\bigartm\build\BigARTM.sln
