#!/usr/bin/env bash
mkdir -p build
cd build
cmake ..
make -j
./train_recognizer ../data/criminals ../data/haarcascades/haarcascade_frontalface_default.xml
Make it executable: chmod +x scripts/run_train.sh
