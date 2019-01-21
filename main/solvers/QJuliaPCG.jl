module QJuliaPCG

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
rdot     = QJuliaReduce.reDotProduct
cpy      = QJuliaBlas.cpy

@inline function MatPrecon(out::AbstractArray, inp::AbstractArray, outSloppy::AbstractArray, inpSloppy::AbstractArray, K::Function)

	if pointer_from_objref(out) == pointer_from_objref(inp); return; end #nothing to do

    outSloppy .=@. 0.0
    cpy(inpSloppy, inp)       #noop for the alias refs
	K(outSloppy, inpSloppy)
    cpy(out, outSloppy)       #noop for the alias refs

end


# nasa2146 matrix norm
const exactAnorm = 3.272816e+07

function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function, extra_args...)

    is_preconditioned = param.inv_type_precondition != QJuliaEnums.QJULIA_INVALID_INVERTER

    solver_name = is_preconditioned == false ? "CG" : "PCG"

    println("Running ", solver_name ," solver.")

    if is_preconditioned == true
      println("Preconditioner  :: ", param.inv_type_precondition)
    end

    if (param.maxiter == 0)
      if param.use_init_guess == false
        x .=@. 0.0
      end
      return
    end #if param.maxiter == 0

    mixed = (param.dtype_sloppy != param.dtype)

    if mixed == true; println("Running mixed precision solver.");end

    local r    = Vector{param.dtype}(undef, length(x))
    local y    = Vector{param.dtype}(undef, length(x))
	# aux high precision vector
    local yaux = ones(param.dtype, length(y))
    # sloppy residual vector
    local rSloppy    = mixed == true ? zeros(param.dtype_sloppy, length(b)) : r
    local rSloppyOld = zeros(param.dtype_sloppy, length(rSloppy))
    # search vector and Ap result
    local p       = zeros(param.dtype_sloppy, length(rSloppy))
    local s       = zeros(param.dtype_sloppy, length(rSloppy))
    # iterated sloppy solution vector
    local xSloppy = param.use_sloppy_partial_accumulator == true ? x : zeros(param.dtype_sloppy, length(rSloppy))
	# for the preconditioner
    local u       = is_preconditioned == true ? zeros(param.dtype_sloppy, length(rSloppy)) : rSloppy
    local rPre    = param.dtype_precondition != param.dtype_sloppy ? zeros(param.dtype_precondition, length(rSloppy)) : rSloppy
    local pPre    = param.dtype_precondition != param.dtype_sloppy ? zeros(param.dtype_precondition, length(rSloppy)) : u

	Precond(out, inp) = MatPrecon(out, inp, pPre, rPre,K)

    b2 = norm2(b)  #Save norm of b
    r2 = 0.0; r2_old = 0.0     #if zero source then we will exit immediately doing no work
    #
	ϵ     = eps(param.dtype_sloppy) / 2.0; ϵh = eps(param.dtype) / 2.0
	deps  = sqrt(ϵ); dfac = 1.1; nfact = 10.0*sqrt(Float64(length(r)))
    xnorm = 0.0; ppnorm = 0.0; Anorm = 0.0
	#
	y .=@. 1.0
	Mat(y, yaux)
	yaux .=@. abs.(y)
	Anorm = findmax(yaux)[1]
	# Relupdates parameters:
    println("Estimated matrix norm is ", Anorm)

    if param.use_init_guess == true
      #r = b - Ax0 <- real
      Mat(r, x)
      r .=@. b - r
	  r2 = norm2(r)
      y .=@. x
    else
      r2 = b2;
      r .=@. b
      y .=@. 0.0
    end
	# Relupdates parameters:
    rUpdate = 0
    rNorm   = sqrt(r2)
    #dinit   = ϵh*(rNorm+nfact*Anorm*xnorm)
	dinit   = ϵh*(rNorm+(nfact+1)*Anorm*xnorm)
    dk      = dinit
    #
	cpy(rSloppy, r)
    #
	Precond(u, rSloppy)
    #
    p  .=@. u; xSloppy .=@. 0.0
    # initialize CG parameters
    α = 0.0; β = 0.0; pAp = 0.0
	#
	γ    = is_preconditioned == true ? rdot(rSloppy, u) : r2
	γold = 0
    #iteration counters
    k = 0; converged = false

    # if invalid residual then convergence is set by iteration count only
    stop = b2*param.tol*param.tol
    println(solver_name," : Initial residual = ", sqrt(r2))

    resIncrease = 0; resIncreaseTotal = 0;  relUpdates = 0

    updateR::Bool = false

    # Main loop:
    while (k < param.maxiter && converged == false)
      # Update search vector
      p  .=@. u + β*p
      #
      MatSloppy(s, p)
      #
      γold = γ
      #
      pAp = rdot(p, s); ppnorm = norm2(p)
      #
      α = γ / pAp
      #
	  xSloppy    .=@. xSloppy + α*p
	  #
      rSloppyOld .=@. rSloppy
      rSloppy    .=@. rSloppy - α*s
      rSloppyOld .=@. rSloppy - rSloppyOld

	  # preconditioned residual
      Precond(u, rSloppy)
      #
	  r2_old = r2
	  # compute remaining dot products
	  r2   = norm2(rSloppy)
      γ    = rdot(u, rSloppy)
      γaux = rdot(u, rSloppyOld)
      γnew = γaux >= 0.0 ? γaux : γ
			#
      β    = γnew / γold
      #
      rNorm   = sqrt(r2)
      # xSloppy .= xSloppy + α*p <=> norm2(xSloppy) = norm2(xSloppy) + α*α*norm2(p)
      xnorm = xnorm + α*α*ppnorm
      dkm1  = dk
      dk    = dkm1 + ϵ*rNorm+ϵh*nfact*Anorm*sqrt(xnorm)
	  #dk  = dkm1 + ϵ*(rNorm + Anorm*sqrt(xnorm) + (nfact+4.0)*abs(α)*Anorm*sqrt(ppnorm))

      updateR =  ( ((dkm1 <= deps*sqrt(γold)) && ((dk > (deps * rNorm)))) && (dk > dfac * dinit) )
	  #updateR = ( (dkm1 <= (deps*sqrt(r2_old))) && ((dk > (deps * rNorm))) )

      if updateR == true
        println("Do reliable update.")
        x .=@. xSloppy
        y .=@. x + y
        Mat(r, y)
        r .=@. b - r
		# Reset sloppy residual and solution vectors
        cpy(rSloppy, r)
		xSloppy .=@. 0.0
		# preconditioned residual
		Precond(u, rSloppy)
        #
		r2 = norm2(r)
		#
        γ  = rdot(rSloppy, u)
        # Reorthogonalize previous search direction against the residual vector
        rp = rdot(rSloppy, p) / r2
        p  .=@. p - rp*rSloppy
        # Recompute β after reliable update
        β  = γ / γold
		# Reset reliable parameters
        dinit = ϵh*(sqrt(r2) + (nfact+1)*Anorm*sqrt(norm2(y)))
        dk = dinit; xnorm = 0.0

        if(sqrt(γ) > rNorm)
          resIncrease      += 1
          resIncreaseTotal += 1
          println("Update residual is higher than iterative residual..", sqrt(γ), " the previous iter residual is ", rNorm)
        else
          resIncrease       = 0
        end

        relUpdates += 1
      end
      # Check convergence:
      converged = (γ > stop) ? false : true
      # Update iter index
      k += 1

     @printf("%s: %d iteration, iter residual: %le \n", solver_name, k, sqrt(γ))

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

end #QJuliaPCG
