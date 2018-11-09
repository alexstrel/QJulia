#load path to qjulia home directory
push!(LOAD_PATH, ENV["QJULIA_HOME"])
push!(LOAD_PATH, string(ENV["QJULIA_HOME"],"/quda-routines"))

import QJuliaBlas
import QJuliaUtils
import QJuliaEnums
import QJuliaInterface
import QJuliaGaugeUtils
import QJuliaComms
import QJuliaSolvers
import QUDARoutines

using Random
using LinearAlgebra

#create function/type alias
m256d   = QJuliaBlas.m256d
m256    = QJuliaBlas.m256
double  = Float64
float   = m256

convert_c2r = QJuliaBlas.convert_c2r

##############################################################################################

[QJuliaUtils.gridsize_from_cmdline[i] = 1 for i = 1:length(QJuliaUtils.gridsize_from_cmdline)]
QJuliaUtils.get_rank_order("col")

#initialize MPI
QJuliaComms.MPI_init_qj(0, C_NULL)

QUDARoutines.initCommsGridQuda_qj(length(QJuliaUtils.gridsize_from_cmdline), QJuliaUtils.gridsize_from_cmdline, QJuliaUtils.lex_rank_from_coords_t_c, C_NULL)

QUDARoutines.initQuda_qj(0)

Random.seed!(2018)

const lx = Cint(16)
const ly = Cint(16)
const lz = Cint(16)
const lt = Cint(32)
const ls = Cint(1 )

const dim = 4
const vol = lx*ly*lz*lt*ls

#field latt point sizes
const ssize = 12 
const gsize = 9  

const splen = vol*ssize
const gflen = vol*gsize

const sp_complex_len = 2*vol*ssize
const sp_complex_parity_len = Int(sp_complex_len / 2)

sp_in = Vector{Complex{Float64}}(undef, splen)
sp_ou = Vector{Complex{Float64}}(undef, splen)
gauge = Matrix{Complex{Float64}}(undef, gflen, 4)

QJuliaUtils.gen_random_spinor!(sp_in)

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

inv_param = QJuliaInterface.QJuliaInvertParam_qj()
inv_param.residual_type = QJuliaEnums.QJULIA_L2_RELATIVE_RESIDUAL
#println("======= Invert parameters =======")
#QJuliaInterface.printQudaInvertParam_qj(inv_param)

QJuliaGaugeUtils.construct_gauge_field!(gauge, 1, gauge_param)

gauge_param.gtype       = QJuliaEnums.QJULIA_SU3_LINKS 		#currently cannot set QJULIA_WILSON_LINKS (=QJULIA_SU3_LINKS)  for QUDA

x_face_size = gauge_param.X[2]*gauge_param.X[3]*Int(gauge_param.X[4]/2);
y_face_size = gauge_param.X[1]*gauge_param.X[3]*Int(gauge_param.X[4]/2);
z_face_size = gauge_param.X[1]*gauge_param.X[2]*Int(gauge_param.X[4]/2);
t_face_size = gauge_param.X[1]*gauge_param.X[2]*Int(gauge_param.X[3]/2);

gauge_param.ga_pad = max(x_face_size, y_face_size, z_face_size, t_face_size);

QUDARoutines.loadGaugeQuda_qj(gauge, gauge_param)

#Check plaquette
plaq = Array{Cdouble, 1}(undef, 3)
QUDARoutines.plaqQuda_qj(plaq)
println("Computed plaquette is ", plaq[1], ", (spatial = ",  plaq[2], ", temporal = ", plaq[3], ")")

mass = -0.9

inv_param.mass = mass
inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass))
inv_param.maxiter = 500

inv_param.cuda_prec = QJuliaEnums.QJULIA_DOUBLE_PRECISION
inv_param.cuda_prec_sloppy = QJuliaEnums.QJULIA_SINGLE_PRECISION
inv_param.cuda_prec_precondition = QJuliaEnums.QJULIA_HALF_PRECISION
inv_param.solution_type = QJuliaEnums.QJULIA_MATPC_SOLUTION 
inv_param.inv_type = QJuliaEnums.QJULIA_PCG_INVERTER

println("Kappa = ",  inv_param.kappa)

matvec(out, inp)    = QUDARoutines.MatDagMatQuda_qj(out, inp, inv_param)
dslash_oe(out, inp) = QUDARoutines.dslashQuda_qj(out, inp, inv_param, QJuliaEnums.QJULIA_EVEN_PARITY)
dslash_eo(out, inp) = QUDARoutines.dslashQuda_qj(out, inp, inv_param, QJuliaEnums.QJULIA_ODD_PARITY )

# Setup preconditioner
precond_param = QJuliaInterface.QJuliaInvertParam_qj()

precond_param.residual_type            = QJuliaEnums.QJULIA_L2_RELATIVE_RESIDUAL
precond_param.inv_type                 = QJuliaEnums.QJULIA_MR_INVERTER
precond_param.dslash_type_precondition = QJuliaEnums.QJULIA_WILSON_DSLASH
precond_param.kappa                    = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass))
precond_param.cuda_prec                = QJuliaEnums.QJULIA_DOUBLE_PRECISION
precond_param.cuda_prec_sloppy         = QJuliaEnums.QJULIA_SINGLE_PRECISION
precond_param.cuda_prec_precondition   = QJuliaEnums.QJULIA_DOUBLE_PRECISION
precond_param.solution_type            = QJuliaEnums.QJULIA_MATPC_SOLUTION 
precond_param.maxiter                  = 6
precond_param.Nsteps    	       = 1

matvecPre(out, inp)  = QUDARoutines.MatDagMatQuda_qj(out, inp, precond_param)

pre_solv_param = QJuliaSolvers.QJuliaSolverParam_qj()

pre_solv_param.inv_type  = precond_param.inv_type
pre_solv_param.tol       = 1e-2
#
pre_solv_param.maxiter   = precond_param.maxiter
pre_solv_param.Nsteps    = 1

K(out, inp) = QJuliaSolvers.solve(out, inp, matvecPre, matvecPre, pre_solv_param)
#K(out, inp) = QUDARoutines.invertQuda_qj(out, inp, precond_param)

x   = Vector{double}(undef, sp_complex_len)
tmp = typeof(x)(undef, length(x))
b   = typeof(x)(undef, length(x))

r   = Vector{double}(undef, sp_complex_parity_len)

x_even = view(x, 1:sp_complex_parity_len)
x_odd  = view(x, sp_complex_parity_len+1:sp_complex_len)

b_even = view(b, 1:sp_complex_parity_len)
b_odd  = view(b, sp_complex_parity_len+1:sp_complex_len)

t_even = view(tmp, 1:sp_complex_parity_len)
t_odd  = view(tmp, sp_complex_parity_len+1:sp_complex_len)

convert_c2r(b, sp_in)

#random intial guess
#QJuliaUtils.gen_random_spinor!(sp_ou, splen)
convert_c2r(x, sp_ou)

#prepare source/solution:
if inv_param.matpc_type == QJuliaEnums.QJULIA_MATPC_EVEN_EVEN
# src = b_e + k D_eo b_o
dslash_eo(t_even, b_odd)
x_odd .=@. b_even + inv_param.kappa*t_even
end

#
x_odd2 = dot(x_odd, x_odd)
#
println("Initial source norm sqrt: ", sqrt(x_odd2), " ,source norm ", x_odd2, " , requested tolerance: ", inv_param.tol)

solv_param = QJuliaSolvers.QJuliaSolverParam_qj()
# Set up parameters
solv_param.inv_type  = inv_param.inv_type
solv_param.tol       = inv_param.tol
#
solv_param.maxiter   = inv_param.maxiter
solv_param.Nsteps    = 2

QJuliaSolvers.solve(x_even, x_odd, matvec, matvec, solv_param, K)

#compute true residual:
matvec(r, x_even)
r  .=@. x_odd - r
r2 = dot(r, r) 
println("True residual sqrt: ", sqrt(r2))

#prepare source/solution:
if inv_param.matpc_type == QJuliaEnums.QJULIA_MATPC_EVEN_EVEN
# x_o = b_o + k D_oe x_e
dslash_oe(t_odd, x_even)
x_odd .=@. b_odd + inv_param.kappa*t_odd
end


QUDARoutines.endQuda_qj()

