#!/usr/bin/env julia

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "..", "matrix"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "core"))

using MatrixBase
using MatrixMarket
using LinearAlgebra
using Random

using QJuliaSolvers
using QJuliaEnums
using MPI

matrix_path = "/home/astrel/data/nasa2146/nasa2146.mtx"
#matrix_path = "/home/astrel/data/nasa2910/nasa2910.mtx"
#matrix_path = "/home/astrel/data/nasa4704/nasa4704.mtx"
#matrix_path = "/home/astrel/data/smt/smt.mtx"

#initialize MPI
MPI.Init()

Random.seed!(2018)

prec        = Float64
prec_sloppy = Float32

csrM       = MatrixBase.CSRMat{prec}(matrix_path)
csrMsloppy = prec_sloppy != prec ? MatrixBase.CSRMat{prec_sloppy}(matrix_path) : csrM

MatrixBase.print_CSRMat_info(csrM)

data_type = typeof(csrM.csrVals[1])

b = Vector{data_type}(undef, csrM.Dims[2])
x = Vector{data_type}(undef, csrM.Dims[2])
z = Vector{data_type}(undef, csrM.Dims[2])

x .= 0.0
z .= 0.0
b .= rand(data_type, csrM.Dims[2])

M(b,x)        = MatrixBase.csrmv(b, csrM, x)
Msloppy(b,x)  = MatrixBase.csrmv(b, csrMsloppy, x)
#or just:
#M=MatrixMarket.mmread(matrix_path)

solv_param = QJuliaSolvers.QJuliaSolverParam_qj()
# Set up parameters
solv_param.dtype                  = prec
solv_param.dtype_sloppy           = prec_sloppy
solv_param.inv_type               = QJuliaEnums.QJULIA_PCG_INVERTER
solv_param.inv_type_precondition  = QJuliaEnums.QJULIA_INVALID_INVERTER
solv_param.tol                    = 1e-8
#
solv_param.maxiter                = 600
solv_param.Nsteps                 = 2

@time QJuliaSolvers.solve(x, b, M, Msloppy, solv_param)

#compute residual:
M(z, x)
z .= b - z

println("Solution norm: ", sqrt(dot(x,x)), ", absolute residual: ", sqrt(dot(z,z)))

MPI.Finalize()
