REM Procedure to generate protobuf files:
REM 1. Copy the following files into $(BIGARTM_ROOT)/src/
REM    - this script
REM    - $(BIGARTM_ROOT)/build/bin/CONFIG/protoc.exe
REM    - $(BIGARTM_ROOT)/build/bin/CONFIG/protoc-gen-cpp_rpcz.exe
REM    Here CONFIG can be either Debug or Release (both options will work equally well).
REM 2. Rename protoc-gen-cpp_rpcz.exe to protoc-gen-rpcz_plugin.exe
REM 3. cd $(BIGARTM_ROOT)/src/
REM 4. run this script.

.\protoc.exe --cpp_out=. --python_out=. .\artm\messages.proto
.\protoc.exe --cpp_out=. --rpcz_plugin_out=. .\artm\core\internals.proto