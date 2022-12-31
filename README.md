# OnnxParser

## How To Generate onnx.proto3.pb.xx
1. Use vcpkg to install protobuf lib
```
vcpkg install protobuf protobuf:x64-windows-static
```
2. Clone onnx from git repo: https://github.com/onnx/onnx
3. Use protobuf to generate onnx.proto3.pb.xx
```
C:\vcpkg\packages\protobuf_x64-windows\tools\protobuf\protoc.exe --cpp_out=. onnx.proto3
```
reference: https://tadaoyamaoka.hatenablog.com/entry/2021/08/18/001934

## What This Tool Does
Parsing .onnx file and getting the network graph