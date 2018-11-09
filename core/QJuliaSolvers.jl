module QJuliaSolvers

  import QJuliaEnums

  function identity_op(out::AbstractArray, inp::AbstractArray)
    out .=@. inp
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

    # Whether to use a pipelined solver with less global sums
    pipeline::Bool
    # Preserve the source or not in the linear solver (deprecated)
    preserve_source::Bool       
    # Whether to use initial guess
    use_init_guess::Bool       

    # The precision used by the QJULIA solver
    precision::DataType               
    # The precision used by the QJULIA sloppy operator
    precision_sloppy::DataType        
    # The precision of the sloppy gauge field for the refinement step in multishift
    precision_refinement_sloppy::DataType
    # The precision used by the QJULIA preconditioner
    precision_precondition::DataType

    # Relaxation parameter used in MR,GCR etc. (default = 1.0)
    omega::Float64
    # Do we want global reductions?
    global_reduction::Bool

    # Number of steps 
    Nsteps::Int

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
				 false,
				 true,
                                 false, 
				 Float64, Float64, Float64, Float64,
				 1.0, 
				 true,
				 0)
				 

  end #QJuliaSolverParam_qj

  include("solvers/QJuliaMR.jl")
  include("solvers/QJuliaPCG.jl")

  function solve(out::AbstractArray, inp::AbstractArray, m::Any, mSloppy::Any, param::QJuliaSolverParam_qj, K::Function = identity_op)

    if param.inv_type == QJuliaEnums.QJULIA_MR_INVERTER
      QJuliaMR.solver(out, inp, m,mSloppy, param)
    elseif param.inv_type == QJuliaEnums.QJULIA_PCG_INVERTER
      QJuliaPCG.solver(out, inp, m,mSloppy, param, K)
    else 
      println("Solver ", param.inv_type," is not available.")
    end 

  end

end #QJuliaSolvers 
