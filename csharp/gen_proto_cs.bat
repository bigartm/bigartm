copy ..\src\artm\messages.proto . /Y
protogen -i:messages.proto -o:messages.cs -p:fixCase
msbuild messages.csproj
