# This is an initial commit

Very brief instructions

1. Current version requires quda dynamic library. This should be compiled with -DQUDA_BUILD_SHAREDLIB=ON cmake option. Of course, one can convert existing static library
into dynamic one. Also set LD_LIBRARY_PATH to directory with libquda.so
I recommend experimental/CAsolvers branch for experiments (use -DQUDA_JULIA=ON cmake option to enable julia binding)
but devel branch should be equally fine. See a WARNING below, though.

WARNING (just if one wants develop quda branch)
I experienced a problem with loading gauge configuration on QUDA. The fix is rather trivial, in the loadGaugeQuda routine one needs explicitely re-assign gauge field pointers, as it was done in lines 714-725 in experimental/CAsolvers branch, please see:
https://github.com/lattice/quda/blob/experimental/CAsolvers/lib/interface_quda.cpp#L714-L725


2. Set path to Julia working directory (modify set_env.sh script, also set up desired number of threads, it's equal to 4 on my machines with for 4 core cpus)

3. Execution is straightforward, just type
julia qjulia_quda_invert_test.jl

4. One needs QUDA_RESOURCE_PATH and QUDA_ENABLE_TUNING env variable, as usual for QUDA.

5. Current verion does not provide command line options, so everything is hard-corded. For example, default is a single-gpu execution. If one wants to execute on many gpus, please modify

QJuliaUtils.gridsize_from_cmdline[i] in line 16 of qjulia_quda_invert_test.jl file.
For example, QJuliaUtils.gridsize_from_cmdline=[1 1 1 4] for 4 gpus etc.

6. Directory structure

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


