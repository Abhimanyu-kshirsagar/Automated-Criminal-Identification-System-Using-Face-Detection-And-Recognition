#!/usr/bin/env bash
mkdir -p build
cd build
cmake ..
make -j
./recognize ../models/lbph_model.yml
