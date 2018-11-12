#!/usr/bin/bash
export QUDA_RESOURCE_PATH=./

qjulia_dir=$(pwd)

export QJULIA_HOME=$qjulia_dir/core

export LD_LIBRARY_PATH=/opt/quda-dyn:$LD_LIBRARY_PATH
export JULIA_NUM_THREADS=4

