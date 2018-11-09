module QJuliaMR

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaSolvers

using LinearAlgebra

norm2 = QJuliaBlas.gnorm2

verbose = false

@inline function cDotProductNormX(x::Vector{T}, y::Vector{T})  where T <: AbstractFloat 
                 global cres = 0.0+0.0im 
                 global rres = 0.0 
                 complex_len = Int(length(x) / 2)
                 for i in 1:complex_len
                    conjcx    = x[2i-1]-x[2i]*im 
                    cy        = y[2i-1]+y[2i]*im 
                    cres += (conjcx * cy)
                    rres += abs2(conjcx)
                 end

                 return (cres , rres) 
end #cDotProductNormX

#First performs the operation y[i] += a*x[i]
#Second performs the operator x[i] -= a*z[i]

@inline function caxpyXmaz(a, x::Vector{T}, y::Vector{T}, z::Vector{T})  where T <: AbstractFloat 

                 complex_len = Int(length(x) / 2)
Threads.@threads for i in 1:complex_len
                    rex = x[2i-1]; imx = x[2i]; rez = z[2i-1]; imz = z[2i]
                    y[2i-1] += (real(a)*rex-imag(a)*imx)
                    y[2i  ] += (real(a)*imx+imag(a)*rex)
                    x[2i-1] -= (real(a)*rez-imag(a)*imz)
                    x[2i  ] -= (real(a)*imz+imag(a)*rez)
                 end
end #caxpyXmaz


function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj) 

    println("Running MR solver in maximum " , param.Nsteps, "steps.")

    if (param.maxiter == 0) || (param.Nsteps == 0) 
      if param.use_init_guess == false 
        x .=@. 0.0 
      end
      return
    end #if

    mixed = (param.precision_sloppy != param.precision)

    global r   = Vector{param.precision}(undef, length(b))
    # now allocate sloppy fields
    global rSloppy = mixed == true ? Vector{param.precision_sloppy}(undef, length(b)) : r  
    global Ar      = typeof(rSloppy)(undef, length(rSloppy))
    #  iterated sloppy solution vector
    global xSloppy = typeof(rSloppy)(undef, length(rSloppy))

    b2 = norm2(b)  #Save norm of b
    r2 = 0.0     #if zero source then we will exit immediately doing no work

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

    # if invalid residual then convergence is set by iteration count only
    stop = b2*param.tol*param.tol
    global step = 0

    println("MR: Initial residual = ", sqrt(r2))

    global converged = false

    while converged == false

      scale = 1.0
      #set to zero sloppy solution
      xSloppy .=@. 0.0

      c2    = param.global_reduction == true ? r2 : norm2(r)  # c2 holds the initial r2
      scale = c2 > 0.0 ? sqrt(c2) : 1.0

      # domain-wise normalization of the initial residual to prevent underflow
      if (c2 > 0.0)
	rSloppy ./=@. scale
	r2 = 1.0 
      end

      global k = 0
      println("MR: ", step, " cycle, ",  k, " iterations, r2 = ", r2)

      while (k < param.maxiter && r2 > 0.0)
        MatSloppy(Ar, rSloppy)

        alpha = cDotProductNormX(Ar, rSloppy)
	# x += omega*alpha*r, r -= omega*alpha*Ar, r2 = blas::norm2(r)//?
        coeff = (param.omega*alpha[1]) / alpha[2]
	caxpyXmaz(coeff, rSloppy, xSloppy, Ar)
	if(verbose == true);println("MR: ", step ," cycle, ", (k+1)," iterations, <r|A|r> = ", real(alpha[1]), ",  ", imag(alpha[1]));end
	
        k += 1
      end #while k < param.maxiter && r2 > 0.0

      # Scale and sum to accumulator
      x .=@. x + scale*xSloppy 

      step += 1 

      if (param.compute_true_res == true || param.Nsteps > 1) 
        Mat(r, x)

        r .=@. b - r 
        r2 = norm2(r)

        param.true_res = sqrt(r2 / b2)

        converged = (step < param.Nsteps && r2 > stop) ? false : true

        if (param.preserve_source == false && converged == true) 
          b .=@. r
        else 
          rSloppy .=@. r
        end 

        println("MR: ", step ," cycle, Converged after ", param.maxiter , "  iterations, relative residual: true = ", sqrt(r2))

      else 
        rSloppy .*= scale

        r2 = norm2(rSloppy)

        converged = (step < param.Nsteps) ? false : true

        if (param.preserve_source == false && converged == true) 
          b .=@. rSloppy
        else 
          r .=@. rSloppy
        end

        println("MR: ", step ," cycle, Converged after ", param.maxiter , "  iterations, relative residual: true = ", sqrt(r2))      

      end #if (param.compute_true_res == true || param.Nsteps > 1) 

    end #while converged == false

end #MR

end #QJuliaMR







