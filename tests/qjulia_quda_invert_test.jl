#!/usr/bin/env julia

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "..", "core"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "libs/quda-routines"))

import QJuliaUtils
import QJuliaEnums
import QJuliaInterface
import QJuliaGaugeUtils
import QJuliaComms
import QUDARoutines

using Random
using MPI

##############################################################################################

[QJuliaUtils.gridsize_from_cmdline[i] = 1 for i = 1:length(QJuliaUtils.gridsize_from_cmdline)]
QJuliaUtils.get_rank_order("col")

#initialize MPI
MPI.Init()

QUDARoutines.initCommsGridQuda_qj(length(QJuliaUtils.gridsize_from_cmdline), QJuliaUtils.gridsize_from_cmdline, QJuliaUtils.lex_rank_from_coords_t_c, C_NULL)

QUDARoutines.initQuda_qj(0)

Random.seed!(2018)

const lx = 16
const ly = 16
const lz = 16
const lt = 32
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

QJuliaUtils.gen_random_spinor!(sp_in)
#print_spinor(sp_in, 4)

gauge_param = QJuliaInterface.QJuliaGaugeParam_qj()

gauge_param.X = (lx, ly, lz, lt)
gauge_param.cpu_prec   = QJuliaEnums.QJULIA_DOUBLE_PRECISION
gauge_param.t_boundary = QJuliaEnums.QJULIA_ANTI_PERIODIC_T
gauge_param.gtype      = QJuliaEnums.QJULIA_WILSON_LINKS
gauge_param.anisotropy = 2.38

gauge_param.cuda_prec                     = QJuliaEnums.QJULIA_DOUBLE_PRECISION
gauge_param.reconstruct                   = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.cuda_prec_sloppy              = QJuliaEnums.QJULIA_HALF_PRECISION
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

QJuliaGaugeUtils.construct_gauge_field!(gauge, 1, gauge_param)

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

mass = -0.9

inv_param.mass = mass
inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass))
inv_param.maxiter = 500
#inv_param.Nsteps = 2
inv_param.tol = 1e-10

inv_param.inv_type = QJuliaEnums.QJULIA_CG_INVERTER
inv_param.cuda_prec = QJuliaEnums.QJULIA_DOUBLE_PRECISION
inv_param.cuda_prec_sloppy = QJuliaEnums.QJULIA_HALF_PRECISION
#inv_param.solve_type = QJuliaEnums.QJULIA_DIRECT_PC_SOLVE
inv_param.solve_type = QJuliaEnums.QJULIA_NORMOP_PC_SOLVE

println("Kappa = ",  inv_param.kappa)

@time QUDARoutines.invertQuda_qj(sp_ou, sp_in, inv_param)

QUDARoutines.endQuda_qj()

MPI.Finalize()
