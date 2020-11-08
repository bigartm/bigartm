REM This script is part of appveyor build. It copies some ARTM files into the installation folder and packs everytying with 7z.

if not exist %INSTALL_FOLDER%\python\examples mkdir %INSTALL_FOLDER%\python\examples
cd C:\projects\bigartm
cp test_data\docword.kos.txt %INSTALL_FOLDER%\python\examples\docword.kos.txt
cp test_data\vocab.kos.txt %INSTALL_FOLDER%\python\examples\vocab.kos.txt
7z.exe a BigARTM.7z %INSTALL_FOLDER%\*
