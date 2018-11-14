module QJuliaPCG

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

    global r   = Vector{param.dtype_sloppy}(undef, length(x))
    # now allocate sloppy fields
    global rSloppy = mixed == true ? Vector{param.dtype_sloppy}(undef, length(b)) : r  
    global p       = typeof(rSloppy)(undef, length(rSloppy))
    global s       = typeof(rSloppy)(undef, length(rSloppy))
    global u       = typeof(rSloppy)(undef, length(rSloppy))
    #  iterated sloppy solution vector
    global xSloppy = typeof(rSloppy)(undef, length(rSloppy))

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

    rSloppy .=@. r

    K(u, rSloppy)
    p  .=@. u

    xSloppy .=@. 0.0

    # if invalid residual then convergence is set by iteration count only
    stop = b2*param.tol*param.tol

    println("PCG: Initial residual = ", sqrt(r2))

    global converged = false

    global ru = rdot(rSloppy, u)
    global ru_old = 0.0

    global k = 0

@time    while (k < param.maxiter && converged == false)

@time      MatSloppy(s, p)
      #
@time      alpha = ru / rdot(s, p)
      # update the residual
@time      rSloppy .=@. rSloppy - alpha*s
      # 
           ru_old = ru
      #
@time      r_newu_old = dot(rSloppy, u)
      # compute precond residual 
@time      K(u, rSloppy)

           ru    = rdot(rSloppy, u)  

           beta  = (ru - r_newu_old) / ru_old

      # update solution and conjugate vector
@time      axpyZpbx(alpha, p, xSloppy, u, beta)

           converged = (ru > stop) ? false : true

#      println("PCG: ", k ," iteration, iter residual: ", sqrt(ru))
           @printf("PCG: %d iteration, iter residual: %le \n", k, sqrt(ru))

           k += 1
         end #while

    x .= @. xSloppy

    if (param.compute_true_res == true) 
      Mat(r, x)

      r .=@. b - r 
      r2 = norm2(r)

      param.true_res = sqrt(r2 / b2)
      println("PCG: converged after ", k , "  iterations, relative residual: true = ", sqrt(r2))

    end #if (param.compute_true_res == true) 

end #solver

end #QJuliaPCG







