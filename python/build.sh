#!/usr/bin/env bash

set -ex

cd $(dirname $(realpath $0))/../build/
cmake .. -DTINYPT_ENABLE_PYBIND=ON
make -j
cp lib/_C.*.so ../python/tinypt/

cd ../python/
pip install .
