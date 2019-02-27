module QJuliaPipeNFCG

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

# References:
# P. Sanan, S.M. Schnepp, and D.A. May, "Pipelined, Flexible Krylov Subspace Methods,"
# SIAM Journal on Scientific Computing 2016 38:5, C441-C470,
# See also:
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

    if param.inv_type_precondition == QJuliaEnums.QJULIA_INVALID_INVERTER
      error("Preconditioner is not defined")
    end

    solver_name = "PipeNFCG"

    mmax = param.nKrylov

    println("Running ", solver_name ," solver (solver precion ", param.dtype, " , sloppy precion ", param.dtype_sloppy, " )")

    if (param.maxiter == 0)
      if param.use_init_guess == false; x .=@. 0.0; end
      return
    end

    mixed = (param.dtype_sloppy != param.dtype)

    if mixed == true; println("Running mixed precision solver."); end

    ϵ = eps(param.dtype_sloppy)
    sqrteps = param.delta*sqrt(ϵ)

    Δcr = 0.0; Δcs = 0.0; Δcw = 0.0; Δcz = 0.0
    errr = 0.0; errrprev = 0.0; errw = 0.0
    errs = zeros(param.dtype, mmax+1)
    errz = zeros(param.dtype, mmax+1)
    replace = 0;totreplaces = 0

    β  = zeros(param.dtype, mmax+1)
    η  = zeros(param.dtype, mmax+1)
    τ  = zeros(param.dtype, mmax+1)

    local r_fp  = zeros(param.dtype, length(x))
    local z_fp  = zeros(param.dtype, length(x))
    local p_fp  = zeros(param.dtype, length(x))
    local w_fp  = zeros(param.dtype, length(x))
    local q_fp  = zeros(param.dtype, length(x))
    local s_fp  = zeros(param.dtype, length(x))
    local u_fp  = zeros(param.dtype, length(x))

    local r   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : r_fp
    local w   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : w_fp
    local u   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : u_fp
    local m   = zeros(param.dtype_sloppy, length(x))
    local n   = zeros(param.dtype_sloppy, length(x))
    local v   = zeros(param.dtype_sloppy, length(x))

    local p   = Matrix{param.dtype_sloppy}(undef, length(x), mmax+1)
    local s   = Matrix{param.dtype_sloppy}(undef, length(x), mmax+1)
    local q   = Matrix{param.dtype_sloppy}(undef, length(x), mmax+1)
    local z   = Matrix{param.dtype_sloppy}(undef, length(x), mmax+1)


    local rPre = zeros(param.dtype_precondition, length(r))
    local pPre = zeros(param.dtype_precondition, length(r))

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
    rnorm  = norm(r_fp)

    Precond(u, r)	    #  u <- Br
    MatSloppy(w, u)		#  w <- Au

    stop = rnorm*rnorm*param.tol*param.tol
    println(solver_name," : Initial (relative) residual ", rnorm / norm2b)

    # zero cycle
    unorm  = norm(u)
    γ      = rdot(r, u)
    δ      = rdot(w, u)
    println(solver_name," : Initial preconditioned residual ", unorm)

    Precond(m, w)	    #   m <- Bw
    MatSloppy(n, m)		#   n <- Am

    η[1]  = δ
    α  = γ / η[1]; β[1] = 0.0

    p[:,1] .=@. u           #  p <- u
    s[:,1] .=@. w           #  s <- w
    q[:,1] .=@. m           #  q <- m
    z[:,1] .=@. n           #  z <- n

    x .=@. x + α*p[:,1]     #  x <- x + alpha * p
    u .=@. u - α*q[:,1]     #  u <- u - alpha * q
    w .=@. w - α*z[:,1]     #  w <- w - alpha * z
    r .=@. r - α*s[:,1]     #  r <- r - alpha * s

    rnorm  = norm(r)
    @printf("%s : first cycle residual: %1.15e \n", solver_name, rnorm/norm2b)

    k = 1; kk = k; converged = false

    Σ  = zeros(param.dtype, mmax+1)
    Ζ  = zeros(param.dtype, mmax+1)

    while (k < param.maxiter && converged == false)

      νi  = max(1, kk%(mmax+1))
      kdx = (kk-1)%(mmax+1)+1

      γold = γ; γ = rdot(r, u)
      for j in (kk-νi):(kk-1) # shift for the fortran-style indexing
        jdx    = j%(mmax+1)+1
        τ[jdx] = rdot(s[:,jdx], u)
      end
      δ     = rdot(w, u)
      unorm = norm(u)

      Σ[kdx]  = sqrt(norm2(s[:,kdx]))
      Ζ[kdx]  = sqrt(norm2(z[:,kdx]))

      v .=@. w - r
      Precond(m, v)		    #   m <- u+B(w-r)
      m .=@. u + m
      MatSloppy(n, m)           #   n <- Am

      kdxp1    = kdx % (mmax+1) + 1
      @printf("\n\nCheck index : i= %d, kdx = %d, kdxp1 = %d (mmax = %d)\n\n", k, kdx, kdxp1, mmax)
      η[kdxp1] = δ
      for j in (kk-νi):(kk-1) # shift for the fortran-style indexing
        jdx    = j%(mmax+1)+1
        β[jdx]    = -τ[jdx] / η[jdx]
        η[kdxp1] -= β[jdx]*β[jdx]*η[jdx]
      end

      αold = α; α = γ / η[kdxp1]

      p[:,kdxp1] .=@. u
      s[:,kdxp1] .=@. w
      q[:,kdxp1] .=@. m
      z[:,kdxp1] .=@. n

      for j in (kk-νi):(kk-1) # shift for the fortran-style indexing
        jdx    = j%(mmax+1)+1
        p[:,kdxp1] .+=@. β[jdx]*p[:,jdx]     #  p <- u + beta * p
        s[:,kdxp1] .+=@. β[jdx]*s[:,jdx]     #  s <- w + beta * s
        q[:,kdxp1] .+=@. β[jdx]*q[:,jdx]     #  q <- m + beta * q
        z[:,kdxp1] .+=@. β[jdx]*z[:,jdx]     #  z <- n + beta * z
      end

      x .=@. x + α*p[:,kdxp1]     #  x <- x + alpha * p
      u .=@. u - α*q[:,kdxp1]     #  u <- u - alpha * q
      w .=@. w - α*z[:,kdxp1]     #  w <- w - alpha * z
      r .=@. r - α*s[:,kdxp1]     #  r <- r - alpha * s

      Δcr = Σ[kdx]; Δcs = αold*Ζ[kdx]; Δcw = αold*Ζ[kdx]; Δcz = 0.0

      for j in (kk-νi):(kk-1) # shift for the fortran-style indexing
        jdx = j%(mmax+1)+1
        Δcs += β[jdx]*Σ[jdx]
        Δcz += β[jdx]*Ζ[jdx]
      end
      Δcr *= (2.0*αold*ϵ)
      Δcs *= (2.0*ϵ)
      Δcw *= (2.0*ϵ)
      Δcz *= (2.0*ϵ)

      if k == 1 || replace == 1
        println("(Re-)initialize reliable parameters..")
        errrprev = errr
        errr = Δcr
        errs[1] = Δcs
        errw = Δcw
        errz[1] = Δcz
        replace = 0
      else
        errrprev = errr
        errr = errr + αold*errs[kdx] + Δcr
        errs[kdxp1] = errw + αold*errz[kdx] + Δcs
        errw = errw + αold*errz[kdx] + Δcw
        errz[kdxp1] = Δcz
        for j in (kk-νi):(kk-1) # shift for the fortran-style indexing
          jdx = j%(mmax+1)+1
          errz[kdxp1] += β[jdx]*errz[jdx]
          errs[kdxp1] += β[jdx]*errs[jdx]
        end
      end

      # Check convergence:
      converged = false # (unorm > stop) ? false : true
      @printf("%s: %d iteration, iter residual: %1.15e\n", solver_name, k, unorm/norm2b)

      if ((k > 1 && errrprev <= (sqrteps * sqrt(γold)) && errr > (sqrteps * sqrt(γ))) || converged == true)
        println("Start reliable update...")
        Mat(r_fp,x)        #  r <- Ax - b
        r_fp .=@. b - r_fp
        norm2r = norm(r_fp)
        cpy(r, r_fp)
        Precond(u,r)     #  u <- Br
        cpy(u_fp,u)
        Mat(w_fp,u_fp)       #  w <- Au
        cpy(w,w_fp)

        #reset parameters
        cpy(p_fp,p[:,kdxp1])
        #
        p .=@. 0.0; s .=@. 0.0;
        z .=@. 0.0; q .=@. 0.0

        Mat(s_fp,p_fp)        #  s <- Ap
        cpy(s[:,1],s_fp)
        Precond(q[:,1],s[:,1])      #  q <- Bs
        cpy(q_fp,q[:,1])
        Mat(z_fp,q_fp)        #  z <- Aq
        cpy(z[:,1],z_fp)
        cpy(p[:,1],p_fp)
        cpy(s[:,1],s_fp)

        @printf("True residual after update %1.15e (relative %1.15e).\n", norm2r, norm2r/norm2b)
        replace = 1;  totreplaces +=1
        # reset internal index
        kk  = 1
      else
        # increment internal index
        kk += 1
      end
      # Update iter index
      k += 1
    end # while
    @printf("Finish %s: %d iterations, total restarst: %d \n", solver_name, k, totreplaces)

end # solver

end # module QJuliaPipeCG
