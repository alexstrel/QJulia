module QJuliaSolvers

  import QJuliaEnums
  import QJuliaBlas

  function identity_op(out::AbstractArray, inp::AbstractArray)
#    out .=@. inp
     QJuliaBlas.cpy(out, inp)
  end


  mutable struct QJuliaSolverParam_qj

    # Which linear solver to use
    inv_type::QJuliaEnums.QJuliaInverterType_qj
    # Which inner solver to use
    inv_type_precondition::QJuliaEnums.QJuliaInverterType_qj

    # Solver tolerance in the L2 residual norm
    tol::Float64
    # Solver tolerance in the L2 residual norm (used to restart InitCG)
    tol_restart::Float64

    # Whether to compute the true residual post solve
    compute_true_res::Bool
    # Actual L2 residual norm achieved in solver
    true_res::Float64
    # Maximum number of iterations in the linear solver
    maxiter::Int
    # Reliable update tolerance
    reliable_delta::Float64
    # Reliable update tolerance used in post multi-shift solver refinement
    reliable_delta_refinement::Float64
    # Whether to keep the partial solution accumuator in sloppy precision
    use_sloppy_partial_accumulator::Bool

    # Whether to use a pipelined solver with less global sums, gives pipeline length
    pipeline::Int
	# Krylov subspace dim
	nKrylov::Int
    # Preserve the source or not in the linear solver (deprecated)
    preserve_source::Bool
    # Whether to use initial guess
    use_init_guess::Bool

    # The precision used by the QJULIA solver
    dtype::DataType
    # The precision used by the QJULIA sloppy operator
    dtype_sloppy::DataType
    # The precision of the sloppy gauge field for the refinement step in multishift
    dtype_refinement_sloppy::DataType
    # The precision used by the QJULIA preconditioner
    dtype_precondition::DataType

    # Relaxation parameter used in MR,GCR etc. (default = 1.0)
    omega::Float64
    # Do we want global reductions?
    global_reduction::Bool

    # Number of steps
    Nsteps::Int

    # Shift
    shift::Float64

    # Reliable updates parameter
    delta::Float64

    #defualt constructor
    QJuliaSolverParam_qj() = new(QJuliaEnums.QJULIA_CG_INVERTER,
                                 QJuliaEnums.QJULIA_INVALID_INVERTER,
				 0.0,
				 0.0,
				 true,
				 0.0,
				 256,
				 1e-1,
				 1e-1,
				 false,
				 0,
				 0,
				 true,
                                 false,
				 Float64, Float64, Float64, Float64,
				 1.0,
				 true,
				 0,
				 0.0,
				 0.01)


  end #QJuliaSolverParam_qj

  include("../main/solvers/QJuliaMR.jl")
  include("../main/solvers/QJuliaLanMR.jl")
  include("../main/solvers/QJuliaPCG.jl")
  include("../main/solvers/QJuliaPipePCG.jl")
  include("../main/solvers/QJuliaPipeCG.jl")
  include("../main/solvers/QJuliaPipeFCG.jl")
  include("../main/solvers/QJuliaCGPCG.jl")

  function solve(out::AbstractArray, inp::AbstractArray, m::Any, mSloppy::Any, param::QJuliaSolverParam_qj, K::Function = identity_op)

    if((K != identity_op) && (param.inv_type_precondition == QJuliaEnums.QJULIA_INVALID_INVERTER))
      error("Preconditioner is not specified..")
    end

    if param.inv_type == QJuliaEnums.QJULIA_MR_INVERTER
      QJuliaMR.solver(out, inp, m,mSloppy, param)
    elseif param.inv_type == QJuliaEnums.QJULIA_LANMR_INVERTER
      QJuliaLanMR.solver(out, inp, m, mSloppy, param)
    elseif param.inv_type == QJuliaEnums.QJULIA_PCG_INVERTER
      QJuliaPCG.solver(out, inp, m, mSloppy, param, K)
    elseif param.inv_type == QJuliaEnums.QJULIA_FCG_INVERTER
      QJuliaFCG.solver(out, inp, m, mSloppy, param, K)
    elseif param.inv_type == QJuliaEnums.QJULIA_PIPEPCG_INVERTER
      #QJuliaCGPCG.solver(out, inp, m,mSloppy, param, K)
	  if K != identity_op
        QJuliaPipePCG.solver(out, inp, m,mSloppy, param, K)
	  else
	    QJuliaPipeCG.solver(out, inp, m,mSloppy, param, K)
	  end
    else
      error("Solver ", param.inv_type," is not available.")
    end

  end

end #QJuliaSolvers
