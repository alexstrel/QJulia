#!/usr/bin/env julia

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "..", "core"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "libs/scidac-routines"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "libs/quda-routines"))

import QJuliaUtils
import QJuliaEnums
import QJuliaInterface
import QJuliaGaugeUtils
import QJuliaComms
import QUDARoutines
import SCIDACRoutines

using Random
using LinearAlgebra
using MPI

##############################################################################################

load_config_from_file = "/home/astrel/Configs/wl_5p5_x2p38_um0p4125_cfg_1000.lime"

[QJuliaUtils.gridsize_from_cmdline[i] = 1 for i = 1:length(QJuliaUtils.gridsize_from_cmdline)]
QJuliaUtils.get_rank_order("col")

#initialize MPI
MPI.Init()

QUDARoutines.initCommsGridQuda_qj(length(QJuliaUtils.gridsize_from_cmdline), QJuliaUtils.gridsize_from_cmdline, QJuliaUtils.lex_rank_from_coords_t_c, C_NULL)

QUDARoutines.initQuda_qj(0)

Random.seed!(2019)

solve_unit_source = true

const lx = 16
const ly = 16
const lz = 16
const lt = 64
const ls = 1

const dim = 4
const vol = lx*ly*lz*lt*ls

#field latt point sizes
const ssize = 12
const gsize = 9

const splen = vol*ssize
const gflen = vol*gsize

sp_in = Vector{Complex{Float64}}(undef, splen)
sp_ou = Vector{Complex{Float64}}(undef, splen)
gauge = Matrix{Complex{Float64}}(undef, gflen, 4)

if solve_unit_source == false
  QJuliaUtils.gen_random_spinor!(sp_in)
else
  QJuliaUtils.gen_unit_spinor!(sp_ou)
end

gauge_param = QJuliaInterface.QJuliaGaugeParam_qj()

gauge_param.X = (lx, ly, lz, lt)
gauge_param.cpu_prec   = QJuliaEnums.QJULIA_DOUBLE_PRECISION
gauge_param.t_boundary = QJuliaEnums.QJULIA_ANTI_PERIODIC_T
gauge_param.gtype      = QJuliaEnums.QJULIA_WILSON_LINKS
gauge_param.anisotropy = 2.38

gauge_param.cuda_prec                     = QJuliaEnums.QJULIA_DOUBLE_PRECISION
gauge_param.reconstruct                   = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.cuda_prec_sloppy              = QJuliaEnums.QJULIA_SINGLE_PRECISION
gauge_param.reconstruct_sloppy            = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.cuda_prec_precondition        = QJuliaEnums.QJULIA_HALF_PRECISION
gauge_param.reconstruct_precondition      = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.reconstruct_refinement_sloppy = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.cuda_prec_refinement_sloppy   = QJuliaEnums.QJULIA_HALF_PRECISION

println("======= Gauge parameters =======")
QUDARoutines.printQudaGaugeParam_qj(gauge_param)

inv_param = QJuliaInterface.QJuliaInvertParam_qj()
inv_param.residual_type = QJuliaEnums.QJULIA_L2_RELATIVE_RESIDUAL

println("======= Invert parameters =======")
QUDARoutines.printQudaInvertParam_qj(inv_param)

gauge_load_type = 1
if load_config_from_file != ""
  Xdims = Vector{Cint}(undef, 4)
  for i in 1:length(Xdims); Xdims[i] = gauge_param.X[i] ; end
  qio_prec = Cint(8) #gauge_param.cuda_prec

  SCIDACRoutines.QMPInitComms_qj(0, C_NULL, QJuliaUtils.gridsize_from_cmdline)
  SCIDACRoutines.read_gauge_field_qj(load_config_from_file, gauge, qio_prec, Xdims, 0, C_NULL)
  gauge_load_type = 2
end
QJuliaGaugeUtils.construct_gauge_field!(gauge, gauge_load_type, gauge_param)

gauge_param.gtype       = QJuliaEnums.QJULIA_SU3_LINKS 		#currently cannot set QJULIA_WILSON_LINKS (=QJULIA_SU3_LINKS)  for QUDA

x_face_size = gauge_param.X[2]*gauge_param.X[3]*Int(gauge_param.X[4]/2);
y_face_size = gauge_param.X[1]*gauge_param.X[3]*Int(gauge_param.X[4]/2);
z_face_size = gauge_param.X[1]*gauge_param.X[2]*Int(gauge_param.X[4]/2);
t_face_size = gauge_param.X[1]*gauge_param.X[2]*Int(gauge_param.X[3]/2);

gauge_param.ga_pad = max(x_face_size, y_face_size, z_face_size, t_face_size)

@time QUDARoutines.loadGaugeQuda_qj(gauge, gauge_param)

#Check plaquette
plaq = Array{Cdouble, 1}(undef, 3)
@time QUDARoutines.plaqQuda_qj(plaq)
println("Computed plaquette is ", plaq[1], ", (spatial = ",  plaq[2], ", temporal = ", plaq[3], ")")

mass = -0.4125
#mass = -0.95

#inv_param.inv_type = QJuliaEnums.QJULIA_CG_INVERTER
inv_param.inv_type = QJuliaEnums.QJULIA_PIPEPCG_INVERTER

inv_param.mass = mass
inv_param.kappa = 1.0 / (2.0 * (1.0 + 3.0/gauge_param.anisotropy + mass))
inv_param.maxiter = 2000
inv_param.Nsteps = 2
inv_param.tol = 1e-9
inv_param.reliable_delta = ( inv_param.inv_type == QJuliaEnums.QJULIA_CG_INVERTER ) ? 1e-2 : 1e+2;

inv_param.cuda_prec = QJuliaEnums.QJULIA_DOUBLE_PRECISION
inv_param.cuda_prec_sloppy = QJuliaEnums.QJULIA_SINGLE_PRECISION
#inv_param.cuda_prec_sloppy = QJuliaEnums.QJULIA_DOUBLE_PRECISION
#inv_param.solve_type = QJuliaEnums.QJULIA_DIRECT_PC_SOLVE
# this are default params but we set it explicitly:
inv_param.solution_type = QJuliaEnums.QJULIA_MAT_SOLUTION
inv_param.solve_type    = QJuliaEnums.QJULIA_NORMOP_PC_SOLVE


# set up the preconditioner:
inv_param.inv_type_precondition  = QJuliaEnums.QJULIA_MR_INVERTER
#inv_param.inv_type_precondition  = QJuliaEnums.QJULIA_INVALID_INVERTER
inv_param.schwarz_type           = QJuliaEnums.QJULIA_ADDITIVE_SCHWARZ
inv_param.precondition_cycle     = 1
inv_param.tol_precondition       = 1e-1
inv_param.maxiter_precondition   = 8
inv_param.cuda_prec_precondition = QJuliaEnums.QJULIA_HALF_PRECISION
inv_param.omega                  = 1.0

inv_param.Nsteps     = 2
inv_param.gcrNkrylov = 8

inv_param.ca_basis      = QJuliaEnums.QJULIA_INVALID_BASIS
inv_param.ca_lambda_min = 0.0
inv_param.ca_lambda_max = 1.0


println("Kappa = ",  inv_param.kappa)

mat(out, inp) = QUDARoutines.MatQuda_qj(out, inp, inv_param)

tmpl_src_norm = norm(sp_ou)

if solve_unit_source == true
  mat(sp_in, sp_ou)
  sp_ou .=@. 0.0
end

init_src_norm = norm(sp_in)
println("Initial source norm:: ", init_src_norm, " , template src norm is: ", tmpl_src_norm)

@time QUDARoutines.invertQuda_qj(sp_ou, sp_in, inv_param)

res_vector = Vector{Complex{Float64}}(undef, splen)
mat(res_vector, sp_ou)
res_vector .=@. sp_in - res_vector
res_norm = norm(res_vector)
println("Residual norm:: ", res_norm)

QUDARoutines.endQuda_qj()

MPI.Finalize()
