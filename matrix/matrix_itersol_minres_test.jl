#!/usr/bin/env julia

#load path to qjulia home directory
push!(LOAD_PATH, string(ENV["QJULIA_HOME"],"/matrix"))

using MatrixBase
using MatrixMarket
using LinearAlgebra
using IterativeSolvers
using LinearMaps
using Random

#matrix_path = "/home/astrel/data/nasa2146/nasa2146.mtx"
matrix_path = "/home/astrel/data/nasa2910/nasa2910.mtx"
#matrix_path = "/home/astrel/data/nasa4704/nasa4704.mtx"
#matrix_path = "/home/astrel/data/smt/smt.mtx"

Random.seed!(2018)

csrM = MatrixBase.CSRMat{Float64}(matrix_path)

MatrixBase.print_CSRMat_info(csrM)

data_type = typeof(csrM.csrVals[1])

b = Vector{data_type}(undef, csrM.Dims[2])
x = Vector{data_type}(undef, csrM.Dims[2])
z = Vector{data_type}(undef, csrM.Dims[2])

x .= 0.0
z .= 0.0
b .= rand(data_type, csrM.Dims[2])

M(v::Vector{data_type}) = MatrixBase.csrmv(csrM, v)

A = LinearMaps.LinearMap{data_type}(M, nothing, csrM.Dims[1], csrM.Dims[2]; issymmetric=true )
#or just:
#M=MatrixMarket.mmread(matrix_path)

@time x, ch = IterativeSolvers.minres!(x, A, b; maxiter=100,verbose=true, log=true)

println(ch)

#compute residual:
z=M(x)
z .= b - z

println("Solution norm: ", sqrt(dot(x,x)), ", absolute residual: ", sqrt(dot(z,z)))
