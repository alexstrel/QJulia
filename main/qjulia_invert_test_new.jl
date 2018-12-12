#load path to qjulia home directory
push!(LOAD_PATH, string(ENV["QJULIA_HOME"],"/core"))
push!(LOAD_PATH, string(ENV["QJULIA_HOME"],"/core/quda-routines"))
push!(LOAD_PATH, string(ENV["QJULIA_HOME"],"/fields"))

import QJuliaFields
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

#create function/type alias
double  = Float64
float   = Float32

##############################################################################################

[QJuliaUtils.gridsize_from_cmdline[i] = 1 for i = 1:length(QJuliaUtils.gridsize_from_cmdline)]
QJuliaUtils.get_rank_order("col")

#initialize MPI
MPI.Init()

QUDARoutines.initCommsGridQuda_qj(length(QJuliaUtils.gridsize_from_cmdline), QJuliaUtils.gridsize_from_cmdline, QJuliaUtils.lex_rank_from_coords_t_c, C_NULL)

QUDARoutines.initQuda_qj(0)

Random.seed!(2019)

const lx = 16
const ly = 16
const lz = 16
const lt = 32
const ls = 1

spinor_field_desc = QJuliaFields.QJuliaLatticeFieldDescr_qj{ComplexF64}(QJuliaEnums.QJULIA_SCALAR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY, false, 0, (lx,ly,lz,lt))

cs_in = QJuliaFields.CreateColorSpinor(spinor_field_desc, 4)
cs_ou = QJuliaFields.CreateColorSpinor(spinor_field_desc, 4)

QJuliaUtils.gen_random_spinor!(cs_in)

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
gauge_param.cuda_prec_precondition        = QJuliaEnums.QJULIA_DOUBLE_PRECISION
gauge_param.reconstruct_precondition      = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.reconstruct_refinement_sloppy = QJuliaEnums.QJULIA_RECONSTRUCT_12
gauge_param.cuda_prec_refinement_sloppy   = QJuliaEnums.QJULIA_HALF_PRECISION

#println("======= Gauge parameters =======")
#QJuliaInterface.printQudaGaugeParam_qj(gauge_param)

gauge_field_desc = QJuliaFields.QJuliaLatticeFieldDescr_qj{ComplexF64}(QJuliaEnums.QJULIA_VECTOR_GEOMETRY, QJuliaEnums.QJULIA_INVALID_PARITY, false, 0, (lx,ly,lz,lt))

gauge_field = QJuliaFields.CreateGaugeField(gauge_field_desc)
QJuliaGaugeUtils.construct_gauge_field!(gauge_field, 1, gauge_param)

inv_param = QJuliaInterface.QJuliaInvertParam_qj()
inv_param.residual_type = QJuliaEnums.QJULIA_L2_RELATIVE_RESIDUAL
#println("======= Invert parameters =======")
#QJuliaInterface.printQudaInvertParam_qj(inv_param)

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
inv_param.maxiter = 100

inv_param.cuda_prec = QJuliaEnums.QJULIA_DOUBLE_PRECISION
inv_param.cuda_prec_sloppy = QJuliaEnums.QJULIA_SINGLE_PRECISION
inv_param.cuda_prec_precondition = QJuliaEnums.QJULIA_HALF_PRECISION
inv_param.solution_type = QJuliaEnums.QJULIA_MATPC_SOLUTION 
#inv_param.inv_type = QJuliaEnums.QJULIA_PIPEPCG_INVERTER
inv_param.inv_type = QJuliaEnums.QJULIA_PCG_INVERTER

println("Kappa = ",  inv_param.kappa)

matvec(out, inp)    = QUDARoutines.MatDagMatQuda_qj(out, inp, inv_param)
dslash_oe(out, inp) = QUDARoutines.dslashQuda_qj(out, inp, inv_param, QJuliaEnums.QJULIA_EVEN_PARITY)
dslash_eo(out, inp) = QUDARoutines.dslashQuda_qj(out, inp, inv_param, QJuliaEnums.QJULIA_ODD_PARITY )

# Setup preconditioner
precond_param = QJuliaInterface.QJuliaInvertParam_qj()

precond_param.residual_type            = QJuliaEnums.QJULIA_L2_RELATIVE_RESIDUAL
#precond_param.inv_type                 = QJuliaEnums.QJULIA_PCG_INVERTER
precond_param.inv_type                 = QJuliaEnums.QJULIA_MR_INVERTER #wroks for naive and fails for pipelined
precond_param.dslash_type_precondition = QJuliaEnums.QJULIA_WILSON_DSLASH
precond_param.kappa                    = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass))
precond_param.cuda_prec                = QJuliaEnums.QJULIA_DOUBLE_PRECISION
precond_param.cuda_prec_sloppy         = QJuliaEnums.QJULIA_SINGLE_PRECISION
precond_param.cuda_prec_precondition   = QJuliaEnums.QJULIA_DOUBLE_PRECISION
precond_param.solution_type            = QJuliaEnums.QJULIA_MATPC_SOLUTION 
precond_param.maxiter                  = precond_param.inv_type == QJuliaEnums.QJULIA_PCG_INVERTER ? 30 : 6
precond_param.Nsteps    	       = 1

matvecPre(out, inp)  = QUDARoutines.MatDagMatQuda_qj(out, inp, precond_param)

pre_solv_param = QJuliaSolvers.QJuliaSolverParam_qj()

pre_solv_param.inv_type  = precond_param.inv_type
pre_solv_param.tol       = 1e-2
#
pre_solv_param.maxiter   = precond_param.maxiter
pre_solv_param.Nsteps    = 1
pre_solv_param.global_reduction = false

K(out, inp) = QJuliaSolvers.solve(out, inp, matvecPre, matvecPre, pre_solv_param)
#K(out, inp) = QUDARoutines.invertQuda_qj(out, inp, precond_param)

x_even = QJuliaFields.Even(cs_ou)
x_odd  = QJuliaFields.Odd(cs_ou)

b_even = QJuliaFields.Even(cs_in)
b_odd  = QJuliaFields.Odd(cs_in)

#Auxiliary field
tmp = QJuliaFields.CreateColorSpinor(spinor_field_desc, 4)

t_even = QJuliaFields.Even(tmp)
t_odd  = QJuliaFields.Odd(tmp)

#prepare source/solution:
if inv_param.matpc_type == QJuliaEnums.QJULIA_MATPC_EVEN_EVEN
# src = b_e + k D_eo b_o
dslash_eo(t_even.v, b_odd.v)
x_odd.v .=@. b_even.v + inv_param.kappa*t_even.v
end

#
x_odd2 = dot(x_odd.v, x_odd.v)
#
println("Initial source norm sqrt: ", sqrt(real(x_odd2)), " ,source norm ", x_odd2, " , requested tolerance: ", inv_param.tol)

solv_param = QJuliaSolvers.QJuliaSolverParam_qj()
# Set up parameters
solv_param.inv_type               = inv_param.inv_type
solv_param.inv_type_precondition  = precond_param.inv_type
solv_param.tol                    = inv_param.tol
#
solv_param.maxiter                = inv_param.maxiter
solv_param.Nsteps                 = 2
#matrix form
#sol = view(reinterpret(cs_ou.field_desc.prec, cs_ou.v), :, :)
#src = view(reinterpret(cs_in.field_desc.prec, cs_in.v), :, :)
#vector form
sol = view(reinterpret(cs_ou.field_desc.prec, cs_ou.v), :)
src = view(reinterpret(cs_in.field_desc.prec, cs_in.v), :)

QJuliaSolvers.solve(sol, src, matvec, matvec, solv_param, K)

#compute true residual:
r = t_odd.v
matvec(r, x_even.v)
r  .=@. x_odd.v - r
r2 = dot(r, r) 
println("True residual sqrt: ", sqrt(real(r2)))

#reconstruct source/solution:
if inv_param.matpc_type == QJuliaEnums.QJULIA_MATPC_EVEN_EVEN
# x_o = b_o + k D_oe x_e
dslash_oe(t_odd.v, x_even.v)
x_odd.v .=@. b_odd.v + inv_param.kappa*t_odd.v
end


QUDARoutines.endQuda_qj()
MPI.Finalize()
