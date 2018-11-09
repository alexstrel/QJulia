module QJuliaPCG

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaSolvers

using LinearAlgebra
using Printf

norm2 = QJuliaBlas.gnorm2

@inline function axpyZpbx(a, p::Vector{T}, x::Vector{T}, u::Vector{T}, b)  where T <: AbstractFloat 

Threads.@threads for i in 1:length(x)
                    x[i] = x[i]+a*p[i] 
                    p[i] = u[i]+b*p[i] 
                 end
end #axpyZpbx

function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function) 

    println("Running MR solver in maximum " , param.Nsteps, "steps.")

    if (param.maxiter == 0)  
      if param.use_init_guess == false 
        x .=@. 0.0 
      end
      return
    end #if param.maxiter == 0

    mixed = (param.precision_sloppy != param.precision)

    global r   = Vector{param.precision_sloppy}(undef, length(x))
    # now allocate sloppy fields
    global rSloppy = mixed == true ? Vector{param.precision_sloppy}(undef, length(b)) : r  
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

    global ru = dot(rSloppy, u)
    global ru_old = 0.0

    global k = 0

@time    while (k < param.maxiter && converged == false)

@time      MatSloppy(s, p)
      #
@time      alpha = ru / dot(s, p)
      # update the residual
@time      rSloppy .=@. rSloppy - alpha*s
      # 
           ru_old = ru
      #
@time      r_newu_old = dot(rSloppy, u)
      # compute precond residual 
@time      K(u, rSloppy)

           ru    = dot(rSloppy, u)  

           beta  = (ru - r_newu_old) / ru_old

      # update solution and conjugate vector
      #xSloppy .=@. xSloppy + alpha*p  
      #p       .=@. u + beta*p
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







