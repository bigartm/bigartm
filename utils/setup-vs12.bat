REM Copy this file to the root folder of BigARTM project

mkdir build_msvc12_x64
cd build_msvc12_x64
set BOOST_ROOT D:\local\boost_1_57_0
set BOOST_LIBRARYDIR=D:\local\boost_1_57_0\lib64-msvc-12.0
cmake .. -G"Visual Studio 12 Win64"
msbuild /p:Configuration=RelWithDebInfo All_Build.vcxproj
msbuild /p:Configuration=RelWithDebInfo INSTALL.vcxproj
REM msbuild /p:Configuration=Debug All_Build.vcxproj
cd ..

mkdir build_msvc12_x86
cd build_msvc12_x86
set BOOST_ROOT D:\local\boost_1_57_0
set BOOST_LIBRARYDIR=D:\local\boost_1_57_0\lib32-msvc-12.0
cmake .. -G"Visual Studio 12"
msbuild /p:Configuration=RelWithDebInfo All_Build.vcxproj
msbuild /p:Configuration=RelWithDebInfo INSTALL.vcxproj
REM msbuild /p:Configuration=Debug All_Build.vcxproj
cd ..

REM "C:\Program Files (x86)\Microsoft Visual Studio 12.0\Common7\IDE\WDExpress.exe" build_msvc12_x86\BigARTM.sln
