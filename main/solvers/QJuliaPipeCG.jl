module QJuliaPipeCG

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

function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function, extra_args...)

	println("WARNING: this solver is in WIP!")

    solver_name = "PipeCG"

    println("Running ", solver_name ," solver (solver precion ", param.dtype, " , sloppy precion ", param.dtype_sloppy, " )")

    if (param.maxiter == 0)
      if param.use_init_guess == false
        x .=@. 0.0
	  end
	  return
	end #if param.maxiter == 0

	mixed = (param.dtype_sloppy != param.dtype)

	if mixed == true; println("Running mixed precision solver."); end

    ϵ = eps(param.dtype_sloppy)
    sqrteps = param.delta*sqrt(ϵ)

	println("Sloppy precision epsilon ", ϵ)

    Δcr = 0.0; Δcs = 0.0; Δcw = 0.0; Δcz = 0.0
	errr = 0.0; errrprev = 0.0; errs = 0.0; errw = 0.0; errz = 0.0
	replace = 0;totreplaces = 0

    local z   = zeros(param.dtype_sloppy, length(x))
    local p   = zeros(param.dtype_sloppy, length(x))
    local w   = zeros(param.dtype_sloppy, length(x))
    local v   = zeros(param.dtype_sloppy, length(x))
    local r   = zeros(param.dtype_sloppy, length(x))
    local s   = zeros(param.dtype_sloppy, length(x))
    local hr  = mixed == true ? zeros(param.dtype, length(x)) : r;
	local hw  = mixed == true ? zeros(param.dtype, length(x)) : w;
	local hs  = mixed == true ? zeros(param.dtype, length(x)) : s;
	local hz  = mixed == true ? zeros(param.dtype, length(x)) : z;
	local hp  = mixed == true ? zeros(param.dtype, length(x)) : p;

    if param.use_init_guess == true
	  #r = b - Ax
	  Mat(hr, x)
	  hr .=@. b - hr
    else
	  hr .=@. b
    end
	cpy(r, hr)

    norm2b = norm(b)
    rnorm  = norm(r)

    MatSloppy(w, r)		#  w <- Ar

    stop = rnorm*rnorm*param.tol*param.tol
    println(solver_name," : Initial residual = ", rnorm / norm2b)

    # zero cycle
	γ   = rdot(r, r)
	δ   = rdot(w, r)

	MatSloppy(v, w)		#   n <- Aw

	α    = γ / δ
	β    = 0.0

	z .=@. v           #  z <- v
	p .=@. r           #  p <- u
	s .=@. w           #  s <- w
	x .=@. x + α*p     #  x <- x + alpha * p
	w .=@. w - α*z     #  w <- w - alpha * z
	r .=@. r - α*s     #  r <- r - alpha * s

	k = 1; converged = false

    while (k < param.maxiter && converged == false)

	  γold = γ
      γ    = rdot(r, r)
      δ    = rdot(w, r)

      Σ  = sqrt(norm2(s))
  	  Ζ  = sqrt(norm2(z))

  	  MatSloppy(v, w)		#   v <- Aw

	  βold = β; β = γ / γold
	  αold = α; α = γ / (δ - β / αold * γ)

      z .=@. v + β*z     #  z <- v + beta * z
      p .=@. r + β*p     #  p <- u + beta * p
      s .=@. w + β*s     #  s <- w + beta * s
	  #
	  x .=@. x + α*p     #  x <- x + alpha * p
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

	  # Check convergence:
	  converged = (γ > stop) ? false : true

      if ((k > 1 && errrprev <= (sqrteps * sqrt(γold)) && errr > (sqrteps * sqrt(γ))) || converged == true)
		println("Start reliable update...")
        Mat(hr,x)        #  r <- Ax - b
        hr .=@. b - hr
		norm2r = norm(hr)
		cpy(r, hr)

        Mat(hw,hr)        #  w <- Ar
		cpy(hp,p)
		Mat(hs,hp)        #  s <- Ap
        Mat(hz,hs)        #  z <- As

        cpy(w,hw)
        cpy(s,hs)
        cpy(z,hz)

        @printf("True residual after update %1.15e (relative %1.15e).\n", norm2r, norm2r/norm2b)
        replace = 1;  totreplaces +=1
      end

	  rnorm = sqrt(γ)
	  @printf("%s: %d iteration, iter residual: %1.15e \n", solver_name, k, rnorm/norm2b)
	  # Update iter index
	  k += 1
    end # while
	@printf("Finish %s: %d iterations, total restarst: %d \n", solver_name, k, totreplaces)

end # solver

end # module QJuliaPipeCG
