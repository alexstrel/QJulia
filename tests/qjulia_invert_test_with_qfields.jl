#!/usr/bin/env julia

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "..", "core"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "libs/quda-routines"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "main/fields"))

import QJuliaGrid
import QJuliaFields
import QJuliaFieldUtils
import QJuliaBlas
import QJuliaReduce
import QJuliaUtils
import QJuliaEnums
import QJuliaInterface
import QJuliaGaugeUtils
import QJuliaComms
import QJuliaSolvers
import QUDARoutines

using Random
using LinearAlgebra
using MPI

##############################################################################################

[QJuliaUtils.gridsize_from_cmdline[i] = 1 for i = 1:length(QJuliaUtils.gridsize_from_cmdline)]
QJuliaUtils.get_rank_order("col")

#initialize MPI
MPI.Init()

QUDARoutines.initCommsGridQuda_qj(length(QJuliaUtils.gridsize_from_cmdline), QJuliaUtils.gridsize_from_cmdline, QJuliaUtils.lex_rank_from_coords_t_c, C_NULL)

QUDARoutines.initQuda_qj(0)

Random.seed!(2019)

const lx = 32
const ly = 32
const lz = 32
const lt = 32
const ls = 1

data_type = ComplexF32
data_prec_type = ComplexF32
data_sloppy_type = ComplexF32

grid_desc = QJuliaGrid.QJuliaGridDescr_qj{data_type}(QJuliaEnums.QJULIA_CPU_FIELD_LOCATION, 0, (lx,ly,lz,lt,ls))

cs_in = QJuliaFields.CreateColorSpinor(grid_desc, QJuliaEnums.QJULIA_INVALID_PARITY)
cs_ou = QJuliaFields.CreateColorSpinor(grid_desc, QJuliaEnums.QJULIA_INVALID_PARITY)

QJuliaFieldUtils.gen_random_spinor!(cs_in)

gauge_param = QJuliaInterface.QJuliaGaugeParam_qj()
gauge_param.X = (lx, ly, lz, lt)
gauge_param.cpu_prec   = data_type == ComplexF64 ? QJuliaEnums.QJULIA_DOUBLE_PRECISION : QJuliaEnums.QJULIA_SINGLE_PRECISION
gauge_param.t_boundary = QJuliaEnums.QJULIA_ANTI_PERIODIC_T
gauge_param.gtype      = QJuliaEnums.QJULIA_WILSON_LINKS
gauge_param.anisotropy = 2.38

gauge_param.cuda_prec                     = data_type == ComplexF64 ? QJuliaEnums.QJULIA_DOUBLE_PRECISION : QJuliaEnums.QJULIA_SINGLE_PRECISION
gauge_param.reconstruct                   = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.cuda_prec_sloppy              = data_sloppy_type == ComplexF64 ? QJuliaEnums.QJULIA_DOUBLE_PRECISION : QJuliaEnums.QJULIA_SINGLE_PRECISION
gauge_param.reconstruct_sloppy            = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.cuda_prec_precondition        = data_prec_type == ComplexF64 ? QJuliaEnums.QJULIA_DOUBLE_PRECISION : QJuliaEnums.QJULIA_SINGLE_PRECISION
gauge_param.reconstruct_precondition      = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.reconstruct_refinement_sloppy = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.cuda_prec_refinement_sloppy   = QJuliaEnums.QJULIA_HALF_PRECISION

println("======= Gauge parameters =======")
QUDARoutines.printQudaGaugeParam_qj(gauge_param)

gauge_field = QJuliaFields.CreateGaugeField(grid_desc)

QJuliaFieldUtils.construct_gauge_field!(gauge_field, 1, gauge_param)

inv_param = QJuliaInterface.QJuliaInvertParam_qj()
inv_param.residual_type = QJuliaEnums.QJULIA_L2_RELATIVE_RESIDUAL

println("======= Invert parameters =======")
QUDARoutines.printQudaInvertParam_qj(inv_param)

gauge_param.gtype       = QJuliaEnums.QJULIA_SU3_LINKS 		#currently cannot set QJULIA_WILSON_LINKS (=QJULIA_SU3_LINKS)  for QUDA

x_face_size = gauge_param.X[2]*gauge_param.X[3]*Int(gauge_param.X[4]/2);
y_face_size = gauge_param.X[1]*gauge_param.X[3]*Int(gauge_param.X[4]/2);
z_face_size = gauge_param.X[1]*gauge_param.X[2]*Int(gauge_param.X[4]/2);
t_face_size = gauge_param.X[1]*gauge_param.X[2]*Int(gauge_param.X[3]/2);

gauge_param.ga_pad = max(x_face_size, y_face_size, z_face_size, t_face_size);

QUDARoutines.loadGaugeQuda_qj(gauge_field.v, gauge_param)

#Check plaquette
plaq = Array{Cdouble, 1}(undef, 3)
QUDARoutines.plaqQuda_qj(plaq)
println("Computed plaquette is ", plaq[1], ", (spatial = ",  plaq[2], ", temporal = ", plaq[3], ")")

mass = -0.9

inv_param.mass = mass
inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass))
inv_param.maxiter = 500
inv_param.tol = 5e-8

inv_param.cpu_prec = gauge_param.cpu_prec
inv_param.cuda_prec = gauge_param.cuda_prec
inv_param.cuda_prec_sloppy = gauge_param.cuda_prec_sloppy
inv_param.cuda_prec_precondition = QJuliaEnums.QJULIA_HALF_PRECISION
inv_param.solution_type = QJuliaEnums.QJULIA_MATPC_SOLUTION
inv_param.inv_type = QJuliaEnums.QJULIA_PIPEPCG_INVERTER
#inv_param.inv_type = QJuliaEnums.QJULIA_PCG_INVERTER

println("Kappa = ",  inv_param.kappa)

mdagm(out, inp) = QUDARoutines.MatDagMatQuda_qj(out, inp, inv_param)
Doe(out, inp)   = QUDARoutines.dslashQuda_qj(out, inp, inv_param, QJuliaEnums.QJULIA_EVEN_PARITY)
Deo(out, inp)   = QUDARoutines.dslashQuda_qj(out, inp, inv_param, QJuliaEnums.QJULIA_ODD_PARITY )

# Setup preconditioner
precond_param = QJuliaInterface.QJuliaInvertParam_qj()

precond_param.residual_type            = QJuliaEnums.QJULIA_L2_RELATIVE_RESIDUAL
precond_param.inv_type                 = QJuliaEnums.QJULIA_PCG_INVERTER
#precond_param.inv_type                 = QJuliaEnums.QJULIA_LANMR_INVERTER #wroks for naive and fails for pipelined
precond_param.inv_type                 = QJuliaEnums.QJULIA_INVALID_INVERTER
precond_param.dslash_type_precondition = QJuliaEnums.QJULIA_WILSON_DSLASH
precond_param.kappa                    = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass))
precond_param.cuda_prec                = data_prec_type == ComplexF64 ? QJuliaEnums.QJULIA_DOUBLE_PRECISION : QJuliaEnums.QJULIA_SINGLE_PRECISION
precond_param.cuda_prec_sloppy         = precond_param.cuda_prec
precond_param.cuda_prec_precondition   = precond_param.cuda_prec
precond_param.solution_type            = QJuliaEnums.QJULIA_MATPC_SOLUTION
precond_param.maxiter                  = precond_param.inv_type == QJuliaEnums.QJULIA_PCG_INVERTER ? 30 : 6
precond_param.Nsteps    	       = 1

mdagmPre(out, inp)  = QUDARoutines.MatDagMatQuda_qj(out, inp, precond_param)

pre_solv_param = QJuliaSolvers.QJuliaSolverParam_qj()

pre_solv_param.dtype = data_prec_type == ComplexF64 ? Float64 : Float32

pre_solv_param.inv_type  = precond_param.inv_type
pre_solv_param.tol       = 1e-2
#
pre_solv_param.maxiter   = precond_param.maxiter
pre_solv_param.Nsteps    = 1
pre_solv_param.global_reduction = false

K(out, inp) = QJuliaSolvers.solve(out, inp, mdagmPre, mdagmPre, pre_solv_param)

x_even = QJuliaFields.Even(cs_ou)
x_odd  = QJuliaFields.Odd(cs_ou)

b_even = QJuliaFields.Even(cs_in)
b_odd  = QJuliaFields.Odd(cs_in)

#Auxiliary field
tmp = QJuliaFields.CreateColorSpinor(spinor_field_desc; NSpin=4)

t_even = QJuliaFields.Even(tmp)
t_odd  = QJuliaFields.Odd(tmp)

#prepare source/solution:
if inv_param.matpc_type == QJuliaEnums.QJULIA_MATPC_EVEN_EVEN
# src = b_e + k D_eo b_o
Deo(t_even.v, b_odd.v)
x_odd.v .=@. b_even.v + inv_param.kappa*t_even.v
end

#
x_odd2 = dot(x_odd.v, x_odd.v)
#
println("Initial source norm: ", sqrt(real(x_odd2)), " ,source norm2 ", x_odd2, " , requested tolerance: ", inv_param.tol)

solv_param = QJuliaSolvers.QJuliaSolverParam_qj()

solv_param.dtype = data_type == ComplexF64 ? Float64 : Float32
solv_param.dtype_sloppy = data_sloppy_type == ComplexF64 ? Float64 : Float32
solv_param.dtype_precondition = data_prec_type == ComplexF64 ? Float64 : Float32
# Set up parameters
solv_param.inv_type               = inv_param.inv_type
solv_param.inv_type_precondition  = precond_param.inv_type
solv_param.tol                    = inv_param.tol
#
solv_param.maxiter                = inv_param.maxiter
solv_param.Nsteps                 = 1

#matrix form
#sol = view(reinterpret(cs_ou.field_desc.prec, x_even.v), :, :)
#src = view(reinterpret(cs_in.field_desc.prec, x_odd.v), :, :)

#vector form
sol = view(reinterpret(cs_ou.field_desc.prec, x_even.v), :)
src = view(reinterpret(cs_in.field_desc.prec, x_odd.v), :)

if precond_param.inv_type != QJuliaEnums.QJULIA_INVALID_INVERTER
  QJuliaSolvers.solve(sol, src, mdagm, mdagm, solv_param, K)
else
  QJuliaSolvers.solve(sol, src, mdagm, mdagm, solv_param)
end

#compute true residual:
r = t_odd.v
mdagm(r, x_even.v)
r  .=@. x_odd.v - r
r2 = dot(r, r)
println("True residual: ", sqrt(real(r2)))

#reconstruct source/solution:
if inv_param.matpc_type == QJuliaEnums.QJULIA_MATPC_EVEN_EVEN
# x_o = b_o + k D_oe x_e
Doe(t_odd.v, x_even.v)
x_odd.v .=@. b_odd.v + inv_param.kappa*t_odd.v
end


QUDARoutines.endQuda_qj()
MPI.Finalize()
