module QJuliaPipePCG

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaReduce
using QJuliaSolvers

using LinearAlgebra
using Printf

norm2    = QJuliaReduce.gnorm2
axpyZpbx = QJuliaBlas.axpyZpbx
rdot     = QJuliaReduce.reDotProduct


function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function)

    println("Running PCG solver.")

    if (param.maxiter == 0)
      if param.use_init_guess == false
        x .=@. 0.0 
      end
      return
    end #if param.maxiter == 0

    mixed = (param.dtype_sloppy != param.dtype)

    if mixed == true
      println("Mixed types is not supported.")
      x .=@. b
      return
    end

    global r   = Vector{param.dtype_sloppy}(undef, length(x))
    # now allocate sloppy fields
    global p       = typeof(r)(undef, length(r))
    global s       = typeof(r)(undef, length(r))
    global u       = typeof(r)(undef, length(r))
    global w       = typeof(r)(undef, length(r))

    global q       = typeof(r)(undef, length(r))
    global z       = typeof(r)(undef, length(r))

    global m       = typeof(r)(undef, length(r))
    global n       = typeof(r)(undef, length(r))

    b2 = norm2(b)  #Save norm of b
    global r2 = 0.0     #if zero source then we will exit immediately doing no work

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

    K(u, r)

    Mat(w,u)

    # if invalid residual then convergence is set by iteration count only
    stop = b2*param.tol*param.tol

    println("PipePCG: Initial residual = ", sqrt(r2))

    converged = false; k = 0

    γ_old     = 0.0
    γ_new_old = 0.0
    α_old     = 0.0

    while (k < param.maxiter && converged == false)

      γ = rdot(r,u)
      δ = rdot(u,w)

      #K(m,w)
      n .=@. w - r
      K(m, n)
      m .=@. u + m
      Mat(n, m)

      if k > 0
        β = (γ - γ_new_old) / γ_old
        α = γ / (δ-(β*γ) / α_old)
      else
        β  = 0.0
        α = γ / δ
      end
      γ_old = γ
      α_old = α

      z .=@. n + β*z
      q .=@. m + β*q

      p .=@. u + β*p
      s .=@. w + β*s

      x .=@. x + α*p
      r .=@. r - α*s

      γ_new_old = rdot(r, u)

      u .=@. u - α*q
      w .=@. w - α*z

      converged = (γ > stop) ? false : true

      @printf("PipePCG: %d iteration, iter residual sq.: %le \n", k, γ)

      k += 1
    end #while

    if (param.compute_true_res == true)
      Mat(r, x)

      r .=@. b - r
      r2 = norm2(r)

      param.true_res = sqrt(r2 / b2)
      println("PipePCG: converged after ", k , "  iterations, relative residual: true = ", sqrt(r2))

    end #if (param.compute_true_res == true)

end #solver

end #QJuliaPipePCG
