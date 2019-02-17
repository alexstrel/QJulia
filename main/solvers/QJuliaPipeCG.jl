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

@inline function MatPrecon(out::AbstractArray, inp::AbstractArray, outSloppy::AbstractArray, inpSloppy::AbstractArray, K::Function)

	if pointer_from_objref(out) == pointer_from_objref(inp); return; end #nothing to do

	outSloppy .=@. 0.0
	cpy(inpSloppy, inp)       #noop for the alias refs
	K(outSloppy, inpSloppy)
	cpy(out, outSloppy)       #noop for the alias refs

end

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

    rnp = 0.0; pnp = 0.0; snp = 0.0; unp = 0.0; wnp = 0.0; xnp = 0.0; qnp = 0.0; znp = 0.0
    replace = 0;totreplaces = 0

    ϵ = eps(param.dtype)
    sqrteps = sqrt(ϵ)

    δs = 0.0; δz = 0.0; δpp = 0.0; δq = 0.0; δm = 0.0
	errr = 0.0; errrprev = 0.0; errs = 0.0; errw = 0.0; errz = 0.0; errncr = 0.0; errncs = 0.0; errncw = 0.0; errncz = 0.0

    local z   = zeros(param.dtype, length(x))
    local p   = zeros(param.dtype, length(x))
    local w   = zeros(param.dtype, length(x))
    local n   = zeros(param.dtype, length(x))
    local r   = zeros(param.dtype, length(x))
    local s   = zeros(param.dtype, length(x))

    local y   = ones(param.dtype, length(x))

    Precond(out, inp) = MatPrecon(out, inp, out, inp, K)

    if param.use_init_guess == true
	  #r = b - Ax
	  Mat(r, x)
	  r .=@. b - r
    else
	  r .=@. b
    end

    norm2b = norm(b)

    Mat(w, r)		#  w <- Ar

	δp    = norm(r) #
    δb    = norm2b
    rnorm = δp

    # Compute matrix norm infinity : ||v|| = max_i |v_i|, ||A|| = max_i || a_i* ||, maximum row sum
    Mat(s, y)
    y .=@. abs.(s)
    #9.43108354e-01
    Anorm = findmax(y)[1]
	mnz   = 10.0	#must be tunable
    sqn   = mnz*sqrt( Float64( length(b) ) )
    println("Extimated matrix norm: \n", Anorm)

    stop = δb*δb*param.tol*param.tol
    println(solver_name," : Initial residual = ", δp / norm2b)

    # zero cycle
	δp  = norm(r)
	γ   = rdot(r, r)
	δ   = rdot(w, r)

	δx  = sqrt(norm2(x))
	δu  = 0.0#sqrt(norm2(r))
	δw  = sqrt(norm2(w))

	Mat(n, w)		#   n <- Aw

	α    = γ / δ
	β    = 0.0
	γold = γ

	z .=@. n           #  z <- n
	p .=@. r           #  p <- u
	s .=@. w           #  s <- w
	x .=@. x + α*p     #  x <- x + alpha * p
	w .=@. w - α*z     #  w <- w - alpha * z
	r .=@. r - α*s     #  r <- r - alpha * s

	k = 1; converged = false

    while (k < param.maxiter && converged == false)

	  pnp = δpp; snp = δs; qnp = δq; znp = δz
      rnp = δp;  unp = δu; wnp = δw; xnp = δx

  	  δp = norm(r)
      γ  = rdot(r, r)
      δ  = rdot(w, r)

      δs  = sqrt(norm2(s))
  	  δz  = sqrt(norm2(z))
  	  δpp = sqrt(norm2(p))

  	  δx  = sqrt(norm2(x))
  	  δw  = sqrt(norm2(w))

  	  Mat(n, w)           #   n <- Am

	  βold = β
      β = γ / γold
	  αold = α
      α = γ / (δ - β / αold * γ)

      z .=@. n + β*z     #  z <- n + beta * z
      p .=@. r + β*p     #  p <- u + beta * p
      s .=@. w + β*s     #  s <- w + beta * s
	  x .=@. x + α*p     #  x <- x + alpha * p
  	  w .=@. w - α*z     #  w <- w - alpha * z
      r .=@. r - α*s     #  r <- r - alpha * s
      γold = γ

      errncr = sqrt(Anorm*xnp+2.0*Anorm*abs(αold)*δpp+rnp+2.0*abs(αold)*δs)*ϵ
      errncw = sqrt(Anorm*unp+2.0*Anorm*abs(αold)*δq+wnp+2.0*abs(αold)*δz)*ϵ

      if k > 1
        errncs = sqrt(Anorm*unp+2.0*Anorm*abs(βold)*pnp+wnp+2.0*abs(βold)*snp)*ϵ
        errncz = sqrt((sqn+2)*Anorm*δm+2.0*Anorm*abs(βold)*qnp+2.0*abs(βold)*znp)*ϵ
      end

      if k == 1
        errr = sqrt((sqn+1)*Anorm*xnp+δb)*ϵ+sqrt(abs(αold)*sqn*Anorm*δpp)*ϵ+errncr
        errs = sqrt(sqn*Anorm*δpp)*ϵ
        errw = sqrt(sqn*Anorm*unp)*ϵ+sqrt(abs(αold)*sqn*Anorm*δq)*ϵ+errncw
        errz = sqrt(sqn*Anorm*δq)*ϵ
      elseif replace == 1
		println("Replace reliable parameters..")
        errrprev = errr
        errr = sqrt((sqn+1)*Anorm*δx+δb)*ϵ
        errs = sqrt(sqn*Anorm*δpp)*ϵ
        errw = sqrt(sqn*Anorm*δu)*ϵ
        errz = sqrt(sqn*Anorm*δq)*ϵ
        replace = 0
      else
        errrprev = errr
        errr = errr+abs(αold)*abs(βold)*errs+abs(αold)*errw+errncr+abs(αold)*errncs
        errs = errw+abs(βold)*errs+errncs
        errw = errw+abs(αold)*abs(βold)*errz+errncw+abs(αold)*errncz
        errz = abs(βold)*errz+errncz
      end

	  converged = (γ > stop) ? false : true
      norm2r = 0.0

      if ((k > 1 && errrprev <= (sqrteps * rnp) && errr > (sqrteps * δp)) || converged == true)
		println("Start reliable update...")
        Mat(r,x)        #  r <- Ax - b
        r .=@. b - r
		norm2r = norm(r)
        Mat(w,r)        #  w <- Ar
		Mat(s,p)        #  s <- Ap
        Mat(z,s)        #  z <- As
        @printf("True residual after update %1.15e (relative %1.15e).\n", norm2r, norm2r/norm2b)
        replace = 1;  totreplaces +=1
      end
	  # Check convergence:
	  rnorm = δp
	  #converged = converged == true ? ((norm2r > stop) ? false : true) : false;
	  #converged = (γ > stop) ? false : true
	  @printf("%s: %d iteration, iter residual: %1.15e \n", solver_name, k, δp/norm2b)
	  # Update iter index
	  k += 1
    end # while

end # solver

end # module QJuliaPipeCG
