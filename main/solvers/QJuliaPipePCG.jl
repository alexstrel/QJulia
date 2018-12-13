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

    global converged = false

    global k = 0

    global gamma_old = 0.0
    global gamma_new_old = 0.0
    global alpha_old = 0.0

    while (k < param.maxiter && converged == false)

      gamma     = rdot(r,u)
      delta     = rdot(u,w)

      #K(m,w)
      n .=@. w - r 
      K(m, n)
      m .=@. u + m 
      Mat(n, m)  

      if k > 0    
        beta  = (gamma - gamma_new_old) / gamma_old
        alpha = gamma / (delta-(beta*gamma)/alpha_old)
      else
        beta  = 0.0
        alpha = gamma / delta
      end
      gamma_old = gamma
      alpha_old = alpha

      z .=@. n + beta*z
      q .=@. m + beta*q

      p .=@. u + beta*p
      s .=@. w + beta*s

      x .=@. x + alpha*p
      r .=@. r - alpha*s

      gamma_new_old = rdot(r, u)

      u .=@. u - alpha*q
      w .=@. w - alpha*z

      converged = (gamma > stop) ? false : true

      @printf("PipePCG: %d iteration, iter residual sq.: %le \n", k, gamma)

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


