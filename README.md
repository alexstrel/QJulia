# QJulia 

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

A Julia package for algorithm prototyping for lattice QFT applications. It currently provides with an interface to the QUDA library that allows one to execute typical
LQCD computational kernels on NVIDIA GPUs. 

## Prerequisites

Julia versions 1.0.1 - 1.0.3, shared QUDA library [QUDA](https://github.com/lattice/quda)  built with `-DQUDA_BUILD_SHAREDLIB=ON` cmake option. Required Julia packagesare : `MPI.jl`, `ArgParse.jl` and `LinearAlgebra`. Optional packages are `IterativeSolvers`, `LinearMaps`, `MatrixMarket`. All packages can be installed with the package mager , e.g., `Pkg.add("PackageName")`


## Instructions

Set `LD_LIBRARY_PATH` to directory with `libquda.so`
I recommend experimental/CAsolvers branch for the QUDA-Julia experiments. (Please use `-DQUDA_JULIA=ON` cmake option to enable julia binding)
The current quda devel branch should be fine as well. See a WARNING below, though.

WARNING (just if one wants develop QUDA branch)
I experienced a problem with loading gauge configuration on QUDA. The fix is rather trivial, in the loadGaugeQuda routine one needs explicitely re-assign gauge field pointers, as it was done in lines 714-725 in experimental/CAsolvers branch, please see:
[CAsolvers](https://github.com/lattice/quda/blob/experimental/CAsolvers/lib/interface_quda.cpp#L714-L725)

Set path to Julia working directory (or modify `set_env.sh` script). Also, setup a desired number of threads for the execution on the host.

Execution is straightforward, e.g., 
`julia qjulia_quda_invert_test.jl` or simply `./qjulia_quda_invert_test.jl`

One needs to `QUDA_RESOURCE_PATH` and `QUDA_ENABLE_TUNING` environemnt variables, as usual for QUDA.

Current verion does not provide command line options, so everything is hard-corded. For example, default is a single-gpu execution. If one wants to execute on many gpus, please modify

`QJuliaUtils.gridsize_from_cmdline[i]` in line 16 of `qjulia_quda_invert_test.jl` file.
For example, `QJuliaUtils.gridsize_from_cmdline=[1 1 1 4]` for 4 gpus etc.

## Structure

1. [core](https://github.com/alexstrel/QJulia/tree/master/core) contains interface stuff, that includes :
    * `QJuliaEnums.jl` provides with a module with QUDA-like enum structures.
    * `QJuliaInterface.jl` currently contains mirror definitions of two main QUDA structures: `QudaGaugeParam` and `QudaInvertParam`. 
    * `quda-routines/QUDARoutines.jl` provides with a module with QUDA interface routines.
    * `QJuliaUtils.jl` containes minimal set of helper methods
    * `QJuliaGaugeUtils.jl` containes gauge field generation methods.

 2. [main](https://github.com/alexstrel/QJulia/tree/master/main) contains basic implementation of iterative solvers and LQFT specific structures
    * `main/solvers` currently includes files with PCG, Pipelined PCG and MinRes implementations
    * `main/fields`  provides with implementations of spinor/gauge field objects
 
 3. [matrix](https://github.com/alexstrel/QJulia/tree/master/matrix) contains implementations that allow to call QJulia solvers for sparse matrices stored in the Matrix Market exchage formats, e.g., from [SuiteSparse](https://sparse.tamu.edu) collection.  This functionality requires `MatrixMarket.jl` package (use `Pkg.add("MatrixMarket")`

 4. [tests](https://github.com/alexstrel/QJulia/tree/master/tests) contains examples of QJulia applications.

