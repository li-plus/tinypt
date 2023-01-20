#!/usr/bin/env bash

set -ex

root_dir=$(dirname $(realpath $0))/../

# c++
cd $root_dir/build
mkdir -p output
./bin/pt -device cpu    -num_samples 100   -scene cornell_sphere   -save_path output/cornell_sphere.png
./bin/pt -device cpu    -num_samples 100   -scene cornell_box      -save_path output/cornell_box.png
./bin/pt -device cpu    -num_samples 12    -scene breakfast_room   -save_path output/breakfast_room.png
./bin/pt -device cpu    -num_samples 12    -scene dabrovic_sponza  -save_path output/dabrovic_sponza.png
./bin/pt -device cpu    -num_samples 12    -scene fireplace_room   -save_path output/fireplace_room.png
./bin/pt -device cpu    -num_samples 12    -scene rungholt         -save_path output/rungholt.png
./bin/pt -device cpu    -num_samples 12    -scene living_room      -save_path output/living_room.png
./bin/pt -device cpu    -num_samples 12    -scene salle_de_bain    -save_path output/salle_de_bain.png

# python
cd $root_dir/python
mkdir -p output
python3 -m examples.main    --device cpu    --num-samples 100   --scene cornell_sphere  --save-path output/cornell_sphere.png
python3 -m examples.main    --device cpu    --num-samples 100   --scene cornell_box     --save-path output/cornell_box.png
python3 -m examples.main    --device cpu    --num-samples 12    --scene breakfast_room  --save-path output/breakfast_room.png
python3 -m examples.main    --device cpu    --num-samples 12    --scene dabrovic_sponza --save-path output/dabrovic_sponza.png
python3 -m examples.main    --device cpu    --num-samples 12    --scene fireplace_room  --save-path output/fireplace_room.png
python3 -m examples.main    --device cpu    --num-samples 12    --scene rungholt        --save-path output/rungholt.png
python3 -m examples.main    --device cpu    --num-samples 12    --scene living_room     --save-path output/living_room.png
python3 -m examples.main    --device cpu    --num-samples 12    --scene salle_de_bain   --save-path output/salle_de_bain.png
