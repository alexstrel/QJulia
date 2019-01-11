#!/usr/bin/env julia

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "..", "matrix"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "core"))

using MatrixBase
using MatrixMarket
using LinearAlgebra

using QJuliaSolvers
using QJuliaEnums
using MPI
using Random

#initialize MPI
MPI.Init()

Random.seed!(2018)

#matrix_path = "/home/astrel/data/nasa2146/nasa2146.mtx"
#matrix_path = "/home/astrel/data/nasa2910/nasa2910.mtx"
matrix_path = "/home/astrel/data/nasa4704/nasa4704.mtx"
#matrix_path = "/home/astrel/data/smt/smt.mtx"

csrM = MatrixBase.CSRMat{Float64}(matrix_path)
csrMpre = MatrixBase.CSRMat{Float64}(matrix_path)

ilu0csrM = MatrixBase.ilu0(csrM)

MatrixBase.print_CSRMat_info(ilu0csrM)

data_type = typeof(csrM.csrVals[1])
pre_data_type = typeof(csrMpre.csrVals[1])

b = Vector{data_type}(undef, csrM.Dims[2])
x = Vector{data_type}(undef, csrM.Dims[2])
z = Vector{data_type}(undef, csrM.Dims[2])

x .= 0.0
z .= 0.0
b .= rand(data_type, csrM.Dims[2])

M(b,x) = MatrixBase.csrmv(b, csrM, x)
#or just:
#M=MatrixMarket.mmread(matrix_path)
Mpre(b,z) = MatrixBase.csrmv(b, csrMpre, z)

#setup preconditioner
pre_solv_param = QJuliaSolvers.QJuliaSolverParam_qj()

pre_solv_param.inv_type  = QJuliaEnums.QJULIA_LANMR_INVERTER
pre_solv_param.dtype     = pre_data_type
pre_solv_param.tol       = 1e-2
#
pre_solv_param.maxiter   = 100
pre_solv_param.Nsteps    = 1
pre_solv_param.global_reduction = false

K(pPre, rPre) = QJuliaSolvers.solve(pPre, rPre, Mpre, Mpre, pre_solv_param)

solv_param = QJuliaSolvers.QJuliaSolverParam_qj()
# Set up parameters
solv_param.inv_type               = QJuliaEnums.QJULIA_PCG_INVERTER
solv_param.dtype                  = data_type
solv_param.inv_type_precondition  = pre_solv_param.inv_type
#solv_param.inv_type_precondition  = QJuliaEnums.QJULIA_INVALID_INVERTER
solv_param.dtype_precondition     = pre_solv_param.dtype
solv_param.tol                    = 1e-9
#
solv_param.maxiter                = 10000
solv_param.Nsteps                 = 2

if (solv_param.inv_type_precondition == QJuliaEnums.QJULIA_INVALID_INVERTER)
@time QJuliaSolvers.solve(x, b, M, M, solv_param)
else
@time QJuliaSolvers.solve(x, b, M, M, solv_param, K)
end

#compute absolute residual:
M(z, x)
z .= b - z

println("Solution norm: ", sqrt(dot(x,x)), ", absolute residual: ", sqrt(dot(z,z)))

MPI.Finalize()
