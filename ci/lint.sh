#!/usr/bin/env bash

set -e

root_dir=$(dirname $(realpath $0))/..

# cpp
cd $root_dir
cmake -B build .
cmake --build build --target format

# python
cd $root_dir/python
isort .
black .
