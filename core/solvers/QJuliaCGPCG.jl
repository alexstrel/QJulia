module QJuliaCGPCG

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

    println("Running Chronopoulos Gear PCG solver.")

    if (param.maxiter == 0)  
      if param.use_init_guess == false 
        x .=@. 0.0 
      end
      return
    end #if param.maxiter == 0

    mixed = (param.dtype_sloppy != param.dtype)

    if mixed == true; error("Mixed types is not supported."); end

    global r   = Vector{param.dtype_sloppy}(undef, length(x))
    # now allocate sloppy fields
    global p       = typeof(r)(undef, length(r))
    global s       = typeof(r)(undef, length(r))
    global u       = typeof(r)(undef, length(r))
    global w       = typeof(r)(undef, length(r))

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

    println("CGPCG: Initial residual = ", sqrt(r2))

    global converged = false

    global k = 0

    global gamma_old = 0.0
    global alpha_old = 0.0

    global case = 2
    while (k < param.maxiter && converged == false)

      gamma     = rdot(r,u)
      delta     = rdot(u,w) 

      if k > 0    
        beta  = gamma / gamma_old
        alpha = gamma / (delta-(beta*gamma)/alpha_old)
      else
        beta  = 0.0
        alpha = gamma / delta
      end
      gamma_old = gamma
      alpha_old = alpha

      p .=@. u + beta*p
      s .=@. w + beta*s

      x .=@. x + alpha*p
      r .=@. r - alpha*s

      K(u, r)
      Mat(w, u)      

      converged = (gamma > stop) ? false : true

      @printf("CGPCG: %d iteration, iter residual: %le \n", k, sqrt(gamma))

      k += 1
    end #while

    if (param.compute_true_res == true) 
      Mat(r, x)

      r .=@. b - r 
      r2 = norm2(r)

      param.true_res = sqrt(r2 / b2)
      println("CGPCG: converged after ", k , "  iterations, relative residual: true = ", sqrt(r2))

    end #if (param.compute_true_res == true) 

end #solver


end #QJuliaCGPCG







