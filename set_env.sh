#!/usr/bin/bash
export QUDA_RESOURCE_PATH=./

qjulia_dir=$(pwd)

export QJULIA_HOME=$qjulia_dir

export LD_LIBRARY_PATH=/opt/quda-dyn:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/scidac-gcc-7.3.0-ompi-3.1.2/lib:$LD_LIBRARY_PATH
export JULIA_NUM_THREADS=4

