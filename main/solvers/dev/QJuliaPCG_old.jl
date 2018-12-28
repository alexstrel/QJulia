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
cpy      = QJuliaBlas.gcpy


function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function) 

    is_preconditioned = param.inv_type_precondition != QJuliaEnums.QJULIA_INVALID_INVERTER

    solver_name = is_preconditioned == false ? "CG" : "PCG"

    println("Running ", solver_name ," solver.")

    if (param.maxiter == 0)  
      if param.use_init_guess == false 
        x .=@. 0.0 
      end
      return
    end #if param.maxiter == 0

    mixed = (param.dtype_sloppy != param.dtype)

    local r   = Vector{param.dtype_sloppy}(undef, length(x))
    # now allocate sloppy fields
    local rSloppy = mixed == true ? Vector{param.dtype_sloppy}(undef, length(b)) : r  
    local p       = typeof(rSloppy)(undef, length(rSloppy))
    local s       = typeof(rSloppy)(undef, length(rSloppy))
#    local u       = is_preconditioned == true ? typeof(rSloppy)(undef, length(rSloppy)) : rSloppy
    local u       = typeof(rSloppy)(undef, length(rSloppy))
    #  iterated sloppy solution vector
    local xSloppy = typeof(rSloppy)(undef, length(rSloppy))

    local rPre    = param.dtype_precondition != param.dtype_sloppy ? Vector{param.dtype_precondition}(undef, length(x)) : rSloppy
    local pPre    = param.dtype_precondition != param.dtype_sloppy ? Vector{param.dtype_precondition}(undef, length(x)) : u

    b2 = norm2(b)  #Save norm of b
    local r2 = 0.0     #if zero source then we will exit immediately doing no work

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

    cpy(rPre, rSloppy) #noop for the alias refs
    K(pPre, rPre)      #noop for the alias reference
    cpy(u, pPre)       #noop for the alias refs

    p  .=@. u

    xSloppy .=@. 0.0

    # if invalid residual then convergence is set by iteration count only
    stop = b2*param.tol*param.tol

    println(solver_name," : Initial residual = ", sqrt(r2))

    local converged = false

    local ru = rdot(rSloppy, u)
    local ru_old = 0.0

    local k = 0

@time    while (k < param.maxiter && converged == false)

      MatSloppy(s, p)
      #
      alpha = ru / rdot(s, p)
      # update the residual
      rSloppy .=@. rSloppy - alpha*s
      # 
      ru_old = ru
      #
      r_newu_old = rdot(rSloppy, u)
      # compute precond residual 
      cpy(rPre, rSloppy) #noop for the alias refs
      K(pPre, rPre)      #noop for the alias reference
      cpy(u, pPre)       #noop for the alias refs

      ru    = rdot(rSloppy, u)  

      beta  = (ru - r_newu_old) / ru_old
      # update solution and conjugate vector
      axpyZpbx(alpha, p, xSloppy, u, beta)

      converged = (ru > stop) ? false : true

#      println("PCG: ", k ," iteration, iter residual: ", sqrt(ru))
      @printf("%s: %d iteration, iter residual: %le \n", solver_name, k, sqrt(ru))

      k += 1
    end #while

    x .= @. xSloppy

    if (param.compute_true_res == true) 
      Mat(r, x)

      r .=@. b - r 
      r2 = norm2(r)

      param.true_res = sqrt(r2 / b2)
      println(solver_name, ": converged after ", k , "  iterations, relative residual: true = ", sqrt(r2))

    end #if (param.compute_true_res == true) 

end #solver

end #QJuliaPCG







