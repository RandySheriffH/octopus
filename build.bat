@echo off
if not exist "build" (
  mkdir build
)
pushd build
if not exist "%1" (
  mkdir %1
)
pushd %1
cmake ..\.. -DCMAKE_BUILD_TYPE=%1
cmake --build . --config=%1
popd
popd