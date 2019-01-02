# QJulia 

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

A Julia package for algorithm prototyping for lattice QFT applications. It currently provides with an interface to the QUDA library that allows one to execute typical
LQCD computational kernels on NVIDIA GPUs. 

## Prerequisites

Julia versions 1.0.1 - 1.0.3, shared QUDA library [QUDA](https://github.com/lattice/quda)  built with `-DQUDA_BUILD_SHAREDLIB=ON` cmake option 


## Instructions

Set `LD_LIBRARY_PATH` to directory with `libquda.so`
I recommend experimental/CAsolvers branch for the QUDA-Julia experiments. (Please use `-DQUDA_JULIA=ON` cmake option to enable julia binding)
The current quda devel branch should be fine as well. See a WARNING below, though.

WARNING (just if one wants develop QUDA branch)
I experienced a problem with loading gauge configuration on QUDA. The fix is rather trivial, in the loadGaugeQuda routine one needs explicitely re-assign gauge field pointers, as it was done in lines 714-725 in experimental/CAsolvers branch, please see:
[](https://github.com/lattice/quda/blob/experimental/CAsolvers/lib/interface_quda.cpp#L714-L725)

Set path to Julia working directory (or modify `set_env.sh` script). Also, setup a desired number of threads for the execution on the host.

Execution is straightforward, e.g., 
`julia qjulia_quda_invert_test.jl` or simply `./qjulia_quda_invert_test.jl`

One needs to `QUDA_RESOURCE_PATH` and `QUDA_ENABLE_TUNING` environemnt variables, as usual for QUDA.

Current verion does not provide command line options, so everything is hard-corded. For example, default is a single-gpu execution. If one wants to execute on many gpus, please modify

`QJuliaUtils.gridsize_from_cmdline[i]` in line 16 of `qjulia_quda_invert_test.jl` file.
For example, `QJuliaUtils.gridsize_from_cmdline=[1 1 1 4]` for 4 gpus etc.

## Directory structure

6.1 ./core contains interface stuff, that includes :
    -- QJuliaEnums.jl => module with QUDA-like enum structures.
       Warning: Julia enumarators by design are bite compatible with C/C++ enumarators, howerever they are not  as flexible as C/C++ counterparts. For instance, it is not possible to wrtie something like this:
       @enum QJuliaPrecision_qj begin
         QJULIA_QUARTER_PRECISION = 1
         QJULIA_HALF_PRECISION    = 2
         QJULIA_SINGLE_PRECISION  = 4
         QJULIA_DOUBLE_PRECISION  = 8
         QJULIA_MYFUNNY_PRECISION = QJULIA_HALF_PRECISION <= ERROR! Julia macro does not allow such kind of re-assignment! All field must have different values.
         QJULIA_INVALID_PRECISION = QJULIA_INVALID_ENUM
       end

    -- QJuliaInterface.jl => module that currently contains mirror definitions of two main QUDA structures: QudaGaugeParam and QudaInvertParam. This will be extended
       with Multigrid/Deflation etc. structures as well.

       Warning: Julia structures must be bite-compatible with corresponding QUDA structures. This means that julia structures must have exactly the same content , including matching static array dimentions, as Quda structures. That is, they must be exact mirrors of Quda structures.

    -- quda-routines/QUDARoutines.jl => module with QUDA interface routines.

    -- QJuliaSolvers.jl => module with an interface to Julia (native) solvers, collected in ./core/solvers/ directory.

       Warning: Julia native solvers are experimental, will be extended for multi-process etc.

    -- QJuliaUtils.jl  => module with minimal set of helper methods

    -- QJuliaGaugeUtils.jl => module with gauge field generation methods (similar to tests_utils.cpp in QUDA)

 6.2  ./main contain test applications:
    qjulia_quda_invert_test.jl runs QUDA inverter (CG by default)
    qjulia_invert_test.jl  runs Julia native inverter with QUDA matrix vector operator (PCG solver with MR as a preconditioner)


