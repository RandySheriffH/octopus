#!/bin/bash
if [[ ! -d ./build ]]
then
  mkdir ./build
fi
cd ./build
if [[ ! -d ./$1 ]]
then
  mkdir ./$1
fi
cd ./$1
cmake ../.. -DCMAKE_BUILD_TYPE=$1
cmake --build . --config $1
