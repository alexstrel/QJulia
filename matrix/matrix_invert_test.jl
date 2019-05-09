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

Random.seed!(2019)

matrix_path = "./nasa2146.mtx"
#matrix_path = "./nasa2910.mtx"
#matrix_path = "/home/astrel/data/nasa2146/nasa2146.mtx"
#matrix_path = "/home/astrel/data/nasa2910/nasa2910.mtx"
#matrix_path = "/home/astrel/data/nasa4704/nasa4704.mtx"
#matrix_path = "/home/astrel/data/bcsstk03/bcsstk03.mtx"

csrM = MatrixBase.CSRMat{Float64}(matrix_path)

MatrixBase.print_CSRMat_info(csrM)

data_type = typeof(csrM.csrVals[1])

b = zero(Vector{data_type}(undef, csrM.Dims[2]))
x = zero(Vector{data_type}(undef, csrM.Dims[2]))
z = zero(Vector{data_type}(undef, csrM.Dims[2]))

M(b,x) = MatrixBase.csrmv(b, csrM, x)
#or just:
#M=MatrixMarket.mmread(matrix_path)

solv_param = QJuliaSolvers.QJuliaSolverParam_qj()
# Set up parameters
#solv_param.inv_type               = QJuliaEnums.QJULIA_PCG_INVERTER
solv_param.inv_type               = QJuliaEnums.QJULIA_GMRESDR_INVERTER
solv_param.inv_type_precondition  = QJuliaEnums.QJULIA_INVALID_INVERTER
solv_param.tol                    = 1e-7
#
solv_param.maxiter                = 1000
solv_param.Nsteps                 = 2
solv_param.nKrylov                = 128
solv_param.pipeline               = 2

do_unit_source = true

if do_unit_source == true
  z .=@. 1.0 / sqrt(csrM.Dims[2])
else
  z .= rand(data_type, csrM.Dims[2])
end
#
M(b,z)
#
@time QJuliaSolvers.solve(x, b, M, M, solv_param)
#
#compute error:
z .=@. z - x
println("Solution error:  ", norm(z))
#
#compute residual:
M(z, x)
z .= b - z

println("Solution norm: ", norm(x), ", absolute residual: ", norm(z), " relative ", norm(z) / norm(b))

MPI.Finalize()
