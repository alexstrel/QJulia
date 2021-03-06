module QJuliaMR

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaReduce
using QJuliaSolvers

using LinearAlgebra
using MPI

norm2             = QJuliaReduce.gnorm2
reDotProductNormX = QJuliaReduce.reDotProductNormX
axpyXmaz          = QJuliaBlas.axpyXmaz

verbose           = false

function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj)

    if verbose == true; println("Running MR solver in maximum " , param.Nsteps, " step(s)."); end

    QJuliaReduce.set_blas_global_reduction(param.global_reduction)

    if MPI.Initialized() == false; error("MPI was not inititalized, copy source field to the solution."); end

    if (param.maxiter == 0) || (param.Nsteps == 0)
      if param.use_init_guess == false
        x .=@. 0.0
      end
      return
    end #if

    mixed = (param.dtype_sloppy != param.dtype)

    local r   = Vector{param.dtype}(undef, length(b))
    # now allocate sloppy fields
    local rSloppy = mixed == true ? Vector{param.dtype_sloppy}(undef, length(b)) : r
    local Ar      = typeof(rSloppy)(undef, length(rSloppy))
    #  iterated sloppy solution vector
    local xSloppy = typeof(rSloppy)(undef, length(rSloppy))

    b2 = norm2(b)  #Save norm of b
    r2 = 0.0     #if zero source then we will exit immediately doing no work

    if param.use_init_guess == true
      #r = b - Ax0 <- real
      Mat(r, x)
      r .=@. b - r
      r2 = norm2(r)
    else
      r2 = b2
      r .=@. b
      x .=@. 0.0
    end

    rSloppy .=@. r

    # if invalid residual then convergence is set by iteration count only
    stop = b2*param.tol*param.tol

    if verbose == true;println("MR: Initial residual = ", sqrt(r2));end

    converged = false; step = 0

    while converged == false

      scale = 1.0
      #set to zero sloppy solution
      xSloppy .=@. 0.0

      c2    = r2  # c2 holds the initial r2
      scale = c2 > 0.0 ? sqrt(c2) : 1.0

      # domain-wise normalization of the initial residual to prevent underflow
      if (c2 > 0.0)
	rSloppy ./=@. scale
	r2 = 1.0
      end

      k = 0
      if verbose == true;println("MR: ", step, " cycle, ",  k, " iterations, r2 = ", r2);end

      while (k < param.maxiter && r2 > 0.0)
        MatSloppy(Ar, rSloppy)

        alpha = reDotProductNormX(Ar, rSloppy)
	# x += omega*alpha*r, r -= omega*alpha*Ar, r2 = blas::norm2(r)//?
        coeff = (param.omega*alpha[1]) / alpha[2]
	axpyXmaz(coeff, rSloppy, xSloppy, Ar)
	if(verbose == true); println("MR: ", step ," cycle, ", (k+1)," iterations, <r|A|r> = ", alpha[1], "  residual norm = ", norm2(rSloppy)); end

        k += 1
      end #while k < param.maxiter && r2 > 0.0

      # Scale and sum to accumulator
      x .=@. x + scale*xSloppy

      step += 1

      if (param.compute_true_res == true || param.Nsteps > 1)
        Mat(r, x)

        r .=@. b - r
        r2 = norm2(r)

        param.true_res = sqrt(r2 / b2)

        converged = (step < param.Nsteps && r2 > stop) ? false : true

        if (param.preserve_source == false && converged == true)
          b .=@. r
        else
          rSloppy .=@. r
        end

        println("MR: ", step ," cycle, Converged after ", k , "  iterations, relative residual: true = ", sqrt(r2))

      else
        rSloppy .*= scale

        r2 = norm2(rSloppy)

        converged = (step < param.Nsteps) ? false : true

        if (param.preserve_source == false && converged == true)
          b .=@. rSloppy
        else
          r .=@. rSloppy
        end

        println("MR: ", step ," cycle, Converged after ", k , "  iterations, relative residual: true = ", sqrt(r2))

      end #if (param.compute_true_res == true || param.Nsteps > 1)

    end #while converged == false

    QJuliaReduce.reset_blas_global_reduction()

end #MR

end #QJuliaMR
