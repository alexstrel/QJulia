module QJuliaPipePCG

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaReduce
using QJuliaSolvers

using LinearAlgebra
using Printf

norm2    = QJuliaReduce.gnorm2
rdot     = QJuliaReduce.reDotProduct
cpy      = QJuliaBlas.cpy

# Reference:
# S. Cools, E.F. Yetkin, E. Agullo, L. Giraud, W. Vanroose, "Analyzing the effect of local rounding error
# propagation on the maximal attainable accuracy of the pipelined Conjugate Gradients method",
# SIAM Journal on Matrix Analysis and Applications (SIMAX), 39(1):426–450, 2018.

@inline function MatPrecon(out::AbstractArray, inp::AbstractArray, outSloppy::AbstractArray, inpSloppy::AbstractArray, K::Function)

	if pointer_from_objref(out) == pointer_from_objref(inp); return; end #nothing to do

	outSloppy .=@. 0.0
	cpy(inpSloppy, inp)       #noop for the alias refs
	K(outSloppy, inpSloppy)
	cpy(out, outSloppy)       #noop for the alias refs

  end

function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function, extra_args...)

	is_preconditioned = param.inv_type_precondition != QJuliaEnums.QJULIA_INVALID_INVERTER

    solver_name = is_preconditioned == false ? "PipeCG" : "PipePCG"

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

	if mixed == true; println("Running mixed precision solver."); end

    ϵ = param.delta*eps(param.dtype_sloppy)
    sqrteps = sqrt(ϵ)

    replace = 0;totreplaces = 0
	Δcr = 0.0; Δcs = 0.0; Δcw = 0.0; Δcz = 0.0
    errr = 0.0; errrprev = 0.0; errs = 0.0; errw = 0.0; errz = 0.0

    #full precision fields
	local r_fp   = zeros(param.dtype, length(x))
    #sloppy precision fields
    local r   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : r_fp
    local z   = zeros(param.dtype_sloppy, length(x))
    local p   = zeros(param.dtype_sloppy, length(x))
    local w   = zeros(param.dtype_sloppy, length(x))
    local q   = zeros(param.dtype_sloppy, length(x))
    local u   = zeros(param.dtype_sloppy, length(x))
    local m   = zeros(param.dtype_sloppy, length(x))
    local n   = zeros(param.dtype_sloppy, length(x))
    local s   = zeros(param.dtype_sloppy, length(x))

	local rPre    = zeros(param.dtype_precondition, length(r))
    local pPre    = zeros(param.dtype_precondition, length(r))

    Precond(out, inp) = MatPrecon(out, inp, rPre, pPre, K)

    if param.use_init_guess == true
	  #r = b - Ax
	  Mat(r_fp, x)
	  r_fp .=@. b - r_fp
    else
	  r_fp .=@. b
    end
	cpy(r, r_fp)

    norm2b = norm(b)

    Precond(u, r)	#  u <- Br
    MatSloppy(w, u)		#  w <- Au

	unorm    = norm(u) #

    stop = unorm*unorm*param.tol*param.tol
    println(solver_name," : Initial residual = ", unorm / norm2b)

	γ  = rdot(r, u)
	δ  = rdot(w, u)

	Precond(m, w)		  #   m <- Bw
	MatSloppy(n, m)           #   n <- Am

    α = γ / δ
	β = 0.0
	z .=@. n          #  z <- n
	q .=@. m          #  q <- m
	p .=@. u          #  p <- u
	s .=@. w          #  s <- w

	x .=@. x + α*p     #  x <- x + alpha * p
	u .=@. u - α*q     #  u <- u - alpha * q
	w .=@. w - α*z     #  w <- w - alpha * z
	r .=@. r - α*s     #  r <- r - alpha * s

    k = 1; converged = false

    while (k < param.maxiter && converged == false)

	  γold = γ; γ = rdot(r, u)
      δ     = rdot(w, u)
      unorm = norm(u)

	  Σ  = sqrt(norm2(s))
  	  Ζ  = sqrt(norm2(z))

	  @printf("%s: %d iteration, iter residual: %1.15e \n", solver_name, k, unorm/norm2b)

	  Precond(m, w)		  #   m <- Bw
	  MatSloppy(n, m)           #   n <- Am

      βold = β; β = γ / γold
      αold = α; α = γ / (δ - β / α * γ)

      z .=@. n + β*z     #  z <- n + beta * z
      q .=@. m + β*q     #  q <- m + beta * q
      p .=@. u + β*p     #  p <- u + beta * p
      s .=@. w + β*s     #  s <- w + beta * s

      x .=@. x + α*p     #  x <- x + alpha * p
      u .=@. u - α*q     #  u <- u - alpha * q
	  w .=@. w - α*z     #  w <- w - alpha * z
      r .=@. r - α*s     #  r <- r - alpha * s

	  Δcr = (2.0*αold*Σ)*ϵ
      Δcs = (2.0*β*Σ+2.0*αold*Ζ)*ϵ
	  Δcw = (2.0*αold*Ζ)*ϵ
	  Δcz = (2.0*β*Ζ)*ϵ

      if k == 1 || replace == 1
		println("(Re-)initialize reliable parameters..")
        errrprev = errr
        errr = Δcr
        errs = Δcs
        errw = Δcw
        errz = Δcz
        replace = 0
      else
        errrprev = errr
        errr = errr + αold*errs + Δcr
        errs = β*errs + errw + αold*errz + Δcs
        errw = errw + αold*errz + Δcw
        errz = β*errz + Δcz
      end

      if (k > 1 && errrprev <= (sqrteps * sqrt(γold)) && errr > (sqrteps * sqrt(γ)))
		println("Start reliable update...")
        Mat(r_fp,x)        #  r <- Ax - b
        r_fp .=@. b - r_fp
		rnorm = norm(r_fp)
		cpy(r, r_fp)
        Precond(u, r)   #  u <- Br
        MatSloppy(w,u)        #  w <- Au
        MatSloppy(s,p)        #  s <- Ap
        Precond(q,s)    #  q <- Bs
        MatSloppy(z,q)        #  z <- Aq
        @printf("True residual after update %1.15e (relative %1.15e).\n", rnorm, rnorm/norm2b)
        replace = 1;  totreplaces +=1
      end
	  # Check convergence:
	  converged = (γ > stop) ? false : true
	  # Update iter index
	  k += 1
    end # while

	@printf("Finish %s: %d iterations, total restarst: %d \n", solver_name, k, totreplaces)

end # solver

end # module
