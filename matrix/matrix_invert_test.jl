#load path to qjulia home directory
push!(LOAD_PATH, string(ENV["QJULIA_HOME"],"/matrix"))
push!(LOAD_PATH, string(ENV["QJULIA_HOME"],"/core"))

using MatrixBase
using MatrixMarket
using LinearAlgebra

using QJuliaSolvers
using QJuliaEnums
using MPI

#initialize MPI
MPI.Init()

#matrix_path = "./nasa2146.mtx"    
matrix_path = "./nasa2910.mtx"
#matrix_path = "./nasa4704.mtx"   

csrM = MatrixBase.CSRMat{Float32}(matrix_path)

MatrixBase.print_CSRMat_info(csrM)

data_type = typeof(csrM.csrVals[1])

b = Vector{data_type}(undef, csrM.Dims[2])
x = Vector{data_type}(undef, csrM.Dims[2])
z = Vector{data_type}(undef, csrM.Dims[2])

x .= 0.0
z .= 0.0
b .= rand(data_type, csrM.Dims[2])

M(b,x) = MatrixBase.csrmv(b, csrM, x)
#or just:
#M=MatrixMarket.mmread(matrix_path)

solv_param = QJuliaSolvers.QJuliaSolverParam_qj()
# Set up parameters
solv_param.inv_type               = QJuliaEnums.QJULIA_PCG_INVERTER
solv_param.inv_type_precondition  = QJuliaEnums.QJULIA_INVALID_INVERTER
solv_param.tol                    = 1e-7
#
solv_param.maxiter                = 10000
solv_param.Nsteps                 = 2

@time QJuliaSolvers.solve(x, b, M, M, solv_param)

#compute residual:
M(z, x)
z .= b - z

println("Solution norm: ", sqrt(dot(b,b)), ", absolute residual: ", sqrt(dot(z,z)))

MPI.Finalize()
