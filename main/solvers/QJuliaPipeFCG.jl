module QJuliaPipeFCG

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

    if(param.inv_type_precondition == QJuliaEnums.QJULIA_INVALID_INVERTER)
      error("Preconditioner is not defined")
    end

    solver_name = "PipeFCG"

    println("Running ", solver_name ," solver (solver precion ", param.dtype, " , sloppy precion ", param.dtype_sloppy, " )")

    if (param.maxiter == 0)
      if param.use_init_guess == false
        x .=@. 0.0
      end
      return
    end

    mixed = (param.dtype_sloppy != param.dtype)

    if mixed == true; println("Running mixed precision solver."); end

    ϵ = eps(param.dtype_sloppy)
    sqrteps = param.delta*sqrt(ϵ)

    Δcr = 0.0; Δcs = 0.0; Δcw = 0.0; Δcz = 0.0
    errr = 0.0; errrprev = 0.0; errs = 0.0; errw = 0.0; errz = 0.0
    replace = 0;totreplaces = 0

    local r_fp  = zeros(param.dtype, length(x))
    local z_fp  = zeros(param.dtype, length(x))
    local p_fp  = zeros(param.dtype, length(x))
    local w_fp  = zeros(param.dtype, length(x))
    local q_fp  = zeros(param.dtype, length(x))
    local s_fp  = zeros(param.dtype, length(x))
    local u_fp  = zeros(param.dtype, length(x))

    local r   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : r_fp
    local z   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : z_fp
    local p   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : p_fp
    local w   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : w_fp
    local q   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : q_fp
    local s   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : s_fp
    local u   = mixed == true ? zeros(param.dtype_sloppy, length(x)) : u_fp
    local m   = zeros(param.dtype_sloppy, length(x))
    local n   = zeros(param.dtype_sloppy, length(x))
    local v   = zeros(param.dtype_sloppy, length(x))

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

    Precond(u, r)     #  u <- Br
    MatSloppy(w, u)   #  w <- Au

    stop = rnorm*rnorm*param.tol*param.tol
    println(solver_name," : Initial (relative) residual ", rnorm / norm2b)

    # zero cycle
    unorm  = norm(u)
    γ      = rdot(r, u)
    δ      = rdot(w, u)
    println(solver_name," : Initial preconditioned residual ", unorm)

    Precond(m, w)    #   m <- Bw
    MatSloppy(n, m)  #   n <- Am

    η  = δ
    α  = γ / η; β = 0.0

    z .=@. n           #  z <- n
    q .=@. m           #  q <- m
    p .=@. u           #  p <- u
    s .=@. w           #  s <- w
    x .=@. x + α*p     #  x <- x + alpha * p
    u .=@. u - α*q     #  u <- u - alpha * q
    w .=@. w - α*z     #  w <- w - alpha * z
    r .=@. r - α*s     #  r <- r - alpha * s

    rnorm  = norm(r)
    @printf("%s : first cycle residual: %1.15e \n", solver_name, rnorm/norm2b)

    k = 1; converged = false

    while (k < param.maxiter && converged == false)

      γold = γ; γ = rdot(r, u)
      τ     = rdot(s, u)
      δ     = rdot(w, u)
      unorm = norm(u)

      Σ  = sqrt(norm2(s))
      Ζ  = sqrt(norm2(z))

      v .=@. w - r
      Precond(m, v)		    #   m <- u+B(w-r)
      m .=@. u + m
      MatSloppy(n, m)           #   n <- Am

      β = -τ / η
      η = δ - β*β*η
      αold = α; α = γ / η

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

      # Check convergence:
      converged = false # (unorm > stop) ? false : true
      @printf("%s: %d iteration, iter residual: %1.15e\n", solver_name, k, unorm/norm2b)

      do_restart = (γ < 0.0) || ( (k > 1 && errrprev <= (sqrteps * sqrt(γold)) && errr > (sqrteps * sqrt(abs(γ)))) || converged == true )

      if do_restart == true
        println("Start reliable update...")
        Mat(r_fp,x)        #  r <- Ax - b
        r_fp .=@. b - r_fp
        norm2r = norm(r_fp)
        cpy(r, r_fp)
        Precond(u,r)     #  u <- Br
        cpy(u_fp,u)
        Mat(w_fp,u_fp)       #  w <- Au
        cpy(w,w_fp)

        cpy(p_fp,p)
        Mat(s_fp,p_fp)        #  s <- Ap
        cpy(s,s_fp)
        Precond(q,s)      #  q <- Bs
        cpy(q_fp,q)
        Mat(z_fp,q_fp)        #  z <- Aq
        cpy(z,z_fp)

        γ = γ < 0.0 ? rdot(u_fp, r_fp) : γ

        converged = (norm2r*norm2r > stop) ? false : true

        @printf("True residual after update %1.15e (relative %1.15e, stop criterio %1.15e).\n", norm2r, norm2r/norm2b, stop)
        replace = 1;  totreplaces +=1
      end
      # Update iter index
      k += 1
    end # while
    @printf("Finish %s: %d iterations, total restarst: %d \n", solver_name, k, totreplaces)

end # solver

end # module QJuliaPipeCG
