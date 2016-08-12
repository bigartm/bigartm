REM This script is part of appveyor build. It copies some ARTM files into the installation folder and packs everytying with 7z.

if not exist %INSTALL_FOLDER%\lib mkdir %INSTALL_FOLDER%\lib
if not exist %INSTALL_FOLDER%\python\examples mkdir %INSTALL_FOLDER%\python\examples
cd C:\projects\bigartm
xcopy C:\projects\bigartm\3rdparty\protobuf-3.0.0\python %INSTALL_FOLDER%\protobuf\Python\ /s /e
xcopy C:\projects\bigartm\3rdparty\protobuf-3.0.0\src\google\protobuf\*.proto %INSTALL_FOLDER%\protobuf\src\google\protobuf\ /s
xcopy C:\projects\bigartm\python\artm\wrapper\messages_pb2.py %INSTALL_FOLDER%\python\artm\wrapper\
cp %PROTOC% %INSTALL_FOLDER%\bin\protoc.exe
cp %PROTOC% %INSTALL_FOLDER%\protobuf\src\protoc.exe
cp build\lib\release\artm.lib %INSTALL_FOLDER%\lib\artm.lib
cp build\lib\release\libprotobuf.lib %INSTALL_FOLDER%\lib\libprotobuf.lib
cp test_data\docword.kos.txt %INSTALL_FOLDER%\python\examples\docword.kos.txt
cp test_data\vocab.kos.txt %INSTALL_FOLDER%\python\examples\vocab.kos.txt
7z.exe a BigARTM.7z %INSTALL_FOLDER%\*
