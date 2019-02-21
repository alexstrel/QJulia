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
using Printf

#matrix_path = "/home/astrel/data/bcsstk15/bcsstk15.mtx"
matrix_path = "/home/alex/data/bcsstk03/bcsstk03.mtx"
#matrix_path = "/home/astrel/data/nasa2146/nasa2146.mtx"
#matrix_path = "/home/astrel/data/nasa2910/nasa2910.mtx"
#matrix_path = "/home/astrel/data/nasa4704/nasa4704.mtx"
#matrix_path = "/home/astrel/data/smt/smt.mtx"
#matrix_path = "/home/alex/data/nos4/nos4.mtx"

#initialize MPI
MPI.Init()

Random.seed!(2018)

prec        = Float64
prec_sloppy = Float64

setprecision(BigFloat, 128) #128 bit

csrM       = MatrixBase.CSRMat{prec}(matrix_path)
#MatrixBase.rescale(csrM, 0.5)
csrMsloppy = prec_sloppy != prec ? MatrixBase.CSRMat{prec_sloppy}(matrix_path) : csrM

MatrixBase.print_CSRMat_info(csrM)

data_type = typeof(csrM.csrVals[1])

b = Vector{data_type}(undef, csrM.Dims[2])
x = Vector{data_type}(undef, csrM.Dims[2])
z = Vector{data_type}(undef, csrM.Dims[2])

x .= 0.0
z .= 0.0
#b .= rand(data_type, csrM.Dims[2])

M(b,x)        = MatrixBase.csrmv(b, csrM, x)

if prec_sloppy != prec
  Msloppy(b,x)  = MatrixBase.csrmv(b, csrMsloppy, x)
else
  Msloppy = M
end
#or just:
#M=MatrixMarket.mmread(matrix_path)
# check matrix vector operation:
println("Begin check")
w64 = ones(data_type, csrM.Dims[2])
v64 = ones(data_type, csrM.Dims[2])
data_type_sloppy = typeof(csrMsloppy.csrVals[1])
w32 = ones(data_type_sloppy, csrM.Dims[2])
v32 = ones(data_type_sloppy, csrM.Dims[2])

M(v64, w64)
w64 .=@. abs.(v64)
#9.43108354e-01
Ainfnorm = findmax(w64)[1]
println("\nExtimated matrix inf norm: \n", Ainfnorm)

ϵ64 = eps(data_type)
ϵ32 = eps(data_type_sloppy)

println("Maximum error bounds: ", Ainfnorm*ϵ64, " for FP and ", Ainfnorm*ϵ32, " for SP")
println("Check unit sources.")
w64 .=@. 1.0
v64 .=@. 0.0
M(v64, w64)
w32 .=@. 1.0
v32 .=@. 0.0
Msloppy(v32, w32)

w64 .=@. v64 - v32
@printf("Difference norm %1.10le,\t(%1.10le vs %1.10le)\n", norm(w64), norm(v32), norm(v64))

println("Check random sources.")

w64 .= rand(data_type, csrM.Dims[2])
v64 .=@. 0.0
M(v64, w64)
w32 .=@. w64
v32 .=@. 0.0
Msloppy(v32, w32)

w64 .=@. v64 - v32
@printf("Difference norm %1.10le,\t(%1.10le vs %1.10le)\n", norm(w64), norm(v32), norm(v64))

println("Check random source cast.")

w64 .= rand(data_type, csrM.Dims[2])
v64 .=@. 0.0
w32 .=@. w64
v32 .=@. 0.0
Msloppy(v32, w32)
w64 .=@. w32
M(v64, w64)

w64 .=@. v64 - v32
@printf("Difference norm %1.10le,\t(%1.10le vs %1.10le)\n", norm(w64), norm(v32), norm(v64))


println("Finish check \n")

solv_param = QJuliaSolvers.QJuliaSolverParam_qj()
# Set up parameters
solv_param.dtype                  = prec
solv_param.dtype_sloppy           = prec_sloppy
solv_param.inv_type               = QJuliaEnums.QJULIA_PIPEPCG_INVERTER
#solv_param.inv_type               = QJuliaEnums.QJULIA_PCG_INVERTER
solv_param.inv_type_precondition  = QJuliaEnums.QJULIA_INVALID_INVERTER
solv_param.tol                    = 1e-8
solv_param.delta                  = 1e-3
#
solv_param.maxiter                = 10000
solv_param.Nsteps                 = 2

do_unit_source = true
if do_unit_source == true
  z .=@. 1.0 / sqrt(csrM.Dims[2])
else
  z .= rand(data_type, csrM.Dims[2])
end

M(b,z)

println("Initial residual:  ", norm(b))

@time QJuliaSolvers.solve(x, b, M, Msloppy, solv_param)

#compute error:
z .=@. z - x
println("Solution error:  ", norm(z))

#compute residual:
M(z, x)
z .= b - z

println("Solution norm: ", norm(x), ", absolute residual: ", norm(z), " relative ", norm(z) / norm(b))

MPI.Finalize()
