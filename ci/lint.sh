#!/usr/bin/env bash

set -e

# cpp
root_dir=$(dirname $(realpath $0))/..
cd $root_dir/build
cmake ..
make format

# python
cd $root_dir/python
autopep8 -i -a -r .
