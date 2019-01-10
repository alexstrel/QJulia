module QJuliaMPPCG

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaReduce
using QJuliaSolvers

using LinearAlgebra
using Printf

##########
# Reference: H. Van der Vorst, Q. Ye, "Residual replacement strategies for Krylov subspace iterative methods for the convergence of true residuals", 1999
##########

norm2    = QJuliaReduce.gnorm2
axpyZpbx = QJuliaBlas.axpyZpbx
rdot     = QJuliaReduce.reDotProduct
cpy      = QJuliaBlas.cpy

@inline function MatPrecon(out::AbstractArray, inp::AbstractArray, outSloppy::AbstractArray, inpSloppy::AbstractArray, K::Function )

  outSloppy .=@. 0.0
  cpy(inpSloppy, inp)       #noop for the alias refs
  K(outSloppy, inpSloppy)   #noop for the alias reference
  cpy(out, outSloppy)       #noop for the alias refs

end

function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function, extra_args...)

    is_preconditioned = param.inv_type_precondition != QJuliaEnums.QJULIA_INVALID_INVERTER

    solver_name = is_preconditioned == false ? "CG" : "PCG"

    println("Running ", solver_name ," solver.")

    if is_preconditioned == true
      println("Preconditioner: ", param.inv_type_precondition)
    end

    if (param.maxiter == 0)
      if param.use_init_guess == false
        x .=@. 0.0
      end
      return
    end #if param.maxiter == 0

    mixed = (param.dtype_sloppy != param.dtype)

    local r   = Vector{param.dtype_sloppy}(undef, length(x))
    local y   = Vector{param.dtype_sloppy}(undef, length(x))
    # now allocate sloppy fields
    local rSloppy    = mixed == true ? Vector{param.dtype_sloppy}(undef, length(b)) : r
    local rSloppyOld = typeof(rSloppy)(undef, length(rSloppy))
    # search vector and Ap result
    local p       = typeof(rSloppy)(undef, length(rSloppy))
    local s       = typeof(rSloppy)(undef, length(rSloppy))
    #  iterated sloppy solution vector
    local xSloppy = param.use_sloppy_partial_accumulator == true ? x : typeof(rSloppy)(undef, length(rSloppy))

    b2 = norm2(b)  #Save norm of b
    r2 = 0.0; r2_old = 0.0     #if zero source then we will exit immediately doing no work

    if param.use_init_guess == true
      #r = b - Ax0 <- real
      Mat(r, x)
      r .=@. b - r
      r2 = norm2(r)
      y .=@. x
    else
      r2 = b2
      r .=@. b
      y .=@. 0.0
    end
    #
    x  .=@. 0.0; xSloppy .=@. 0.0
    cpy(rSloppy, r)
    #
    ϵ  = eps(param.dtype_sloppy) / 2.0; ϵh = eps(param.dtype) / 2.0
    #
    deps = sqrt(ϵ); dfac = 1.1
    xnorm = 0.0; ppnorm = 0.0; Anorm = 0.0
    # Estimate A norm:
    Anorm = sqrt(r2 / b2)
    # Initialize search direction:
    if length(extra_args) > 0
      if (typeof(extra_args[1]) == typeof(x)) || (typeof(extra_args[1]) == typeof(xSloppy))
        println("Loading init search vector..")
        p  .=@. extra_args[1]
      end
    else
      p .=@. r
    end
    # initialize CG parameters
    α = 0.0; β = 0.0; pAp = 0.0

    # Initial orthogonalization step:
    if length(extra_args) > 1
      println("Initial orthogonalization step.")
      r2_old = extra_args[2]
      rp = rdot(rSloppy, p) / r2
      β  = r2 / r2_old
      p .=@. rSloppy + β*p
    end

    # Relupdates parameters:
    rUpdate = 0
    rNorm   = sqrt(r2)
    dinit   = ϵh*(rNorm+Anorm*xnorm)
    dk      = dinit

    #iteration counters
    k = 0; converged = false

    # if invalid residual then convergence is set by iteration count only
    stop = b2*param.tol*param.tol
    println(solver_name," : Initial residual = ", sqrt(r2))

    resIncrease = 0; resIncreaseTotal = 0;  relUpdates = 0

    updateR::Bool = false

    # Main loop:
    while (k < param.maxiter && converged == false)
      MatSloppy(s, p)
      #
      r2_old = r2
      #
      pAp = rdot(p, s); ppnorm = rdot(p, p)
      #
      α = r2 / pAp
      #
      rSloppyOld .=@. rSloppy
      rSloppy    .=@. rSloppy - α*s
      xSloppy    .=@. xSloppy + α*p
      rSloppyOld .=@. rSloppy - rSloppyOld
      #
      r2   = norm2(rSloppy)
      γaux = rdot(rSloppy, rSloppyOld)
      γ    = γaux >= 0.0 ? γaux : r2
      β    = γ / r2_old
      #
      rNorm   = sqrt(r2)
      # xSloppy .= xSloppy + α*p <=> norm2(xSloppy) = norm2(xSloppy) + α*α*norm2(p)
      xnorm = xnorm + α*α*ppnorm
      dkm1  = dk
      dk    = dkm1 + ϵ*rNorm+ϵh*Anorm*sqrt(xnorm)

      updateR = ( ((dkm1 <= deps*sqrt(r2_old)) && ((dk > (deps * rNorm)))) && (dk > dfac * dinit) )

      if updateR == true
        println("Do reliable update.")
        x .=@. xSloppy
        y .=@. x + y
        Mat(r, y)
        r .=@. b - r
        r2 = norm2(r)
        cpy(rSloppy, r)
        # Reset sloppy solution vector
        xSloppy .=@. 0.0
        # Reset reliable parameters
        Anorm = sqrt(r2/ b2)
        dinit = ϵh*(sqrt(r2) + Anorm*sqrt(norm2(y)))
        dk = dinit; xnorm = 0.0

        if(sqrt(r2) > rNorm)
          resIncrease      += 1
          resIncreaseTotal += 1
          println("Update residual is higher than iterative residual..", sqrt(r2), " the previous iter residual is ", rNorm)
        else
          resIncrease       = 0
        end
        # Reorthogonalize previous search direction against the residual vector
        rp = rdot(rSloppy, p) / r2
        p  .=@. p - rp*rSloppy
        # Recompute β after reliable update
        β  = r2 / r2_old

        relUpdates += 1
      end

      # Update search vector
      p  .=@. rSloppy + β*p
      # Check convergence:
      converged = (r2 > stop) ? false : true
      # Update iter index
      k += 1

     @printf("%s: %d iteration, iter residual: %le \n", solver_name, k, sqrt(r2))

    end #while

    x .= @. xSloppy
    y .= @. x + y

    if (param.compute_true_res == true)
      Mat(r, y)

      r .=@. b - r
      r2 = norm2(r)

      param.true_res = sqrt(r2 / b2)
      println(solver_name, ": converged after ", k , "  iterations, relative residual: true = ", sqrt(r2), " after reliable updates number ", relUpdates)

    end #if (param.compute_true_res == true)

    x .= @. y
end #solver

end #QJuliaMPPCG
