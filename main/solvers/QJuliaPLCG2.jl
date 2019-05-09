module QJuliaPLCG2

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaReduce
using QJuliaSolvers

using LinearAlgebra
using MPI

norm2   = QJuliaReduce.gnorm2
rdot    = QJuliaReduce.reDotProduct
verbose = true

const ε = eps(Float64)


function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj)

    if verbose == true; println("Running LPCG in maximum " , param.nKrylov, " iters."); end

    if MPI.Initialized() == false; error("MPI was not inititalized, copy source field to the solution."); end

    if param.nKrylov > param.maxiter; println("Warning: search space exceeds requested maximum iter count. Some routines may fail."); end
    if param.pipeline < 1; println("Pipeline length was not set."); end

    if (param.maxiter == 0) || (param.nKrylov == 0)
      if param.use_init_guess == false
        x .=@. 0.0
      end
      return
    end #if

    mixed = (param.dtype_sloppy != param.dtype)

    σ = param.shift
    l = param.pipeline

    nkrylov = param.nKrylov
    zsize   = nkrylov+1+l
    data_type = typeof(x[1]) == BigFloat ? BigFloat : Float64

    local r    = zero(typeof(x)(undef, length(x)))
    local p    = zero(typeof(x)(undef, length(x)))
    # Basis vectors (incl. auxiliary vectors)
    local Z  = zero(Matrix{typeof(x[1])}(undef, length(x), (l+1)*zsize))

    Z1   = view(Z, :, 1:zsize)
    Z2   = view(Z, :, (zsize+1):2zsize)    
    Zlp1 = view(Z, :, (l*zsize+1):(l+1)*zsize)

    # G matrix
    local g = zero(Matrix{data_type}(undef, (nkrylov+1+4l), nkrylov+4l))

    local γ = zero(Vector{data_type}(undef, (nkrylov+1)))
    local δ = zero(Vector{data_type}(undef, (nkrylov+1)))

    bnorm = sqrt(norm2(b))  #Save norm of b
    rnorm = 0.0     #if zero source then we will exit immediately doing no work

    z11 = view(Z1, :, 1)

    if param.use_init_guess == true
      #r = b - Ax0
      Mat(z11, x)
      z11 .=@. z11 - σ*x
      z11 .=@. b - z11
      rnorm = sqrt(norm2(z11))
    else
      rnorm = bnorm
      z11 .=@. b
    end

    if( rnorm == 0.0 ); return; end
    if verbose == true; println("LPCG: Initial residual = ", rnorm); end

    # Scale the first basis vector:
    z11 .=@. z11 / rnorm
    # Inititalize aux basis
    for i in 2:(l+1)
      Zi  = view(Z, :, ((i-1)*zsize+1):i*zsize)
      zi1 = view(Zi,:, 1)
      zi1 .=@. z11
    end

    #
    g[1,1] = 1.0
    # CG params:
    η = 0.0; ζ = 0.0; λ = 0.0

    #start iterations:
    for i in 1:(min(param.maxiter, nkrylov)+l)
      #
      zlp1ip1 = view(Zlp1,:,i+1)
      zlp1i   = view(Zlp1,:,i)
      #
      MatSloppy(zlp1ip1, zlp1i)
      #
      if i <= l; zlp1ip1 .=@. zlp1ip1 - σ * zlp1i; end # apply shift σ -> σ[i]

      if i <= (l-1)
        for k in (i+1):l
          Zk = view(Z, :, ((k-1)*zsize+1):k*zsize)
          zkip1 = view(Zk,:,i+1)
          zkip1 .=@. zlp1ip1
        end
      end

      #
      a  = i - l
      #
      if a >= 1
        for j in max(a-l+2, 1):a
          for k in max(a+1-2l, 1):j-1
            g[j,a+1] -= (g[k,j]*g[k,a+1])
          end
          g[j,a+1] /= g[j,j]
        end
        for k in max(a+1-2l, 1):a
          g[a+1,a+1] -=g[k,a+1]*g[k,a+1]
        end

        if g[a+1,a+1] <= 0.0
          println("SQRT breakdown detected at iteration ", i)
          break
        end
        #
        g[a+1,a+1] = sqrt(g[a+1,a+1])
        if a <= l
          third_term = a == 1 ? 0.0 : δ[a-1]*g[a-1,a]
          #
          γ[a] = (g[a,a+1] + σ*g[a,a] - third_term) / g[a,a] #Note σ -> σ[a]
          δ[a] = g[a+1,a+1] / g[a,a]
        else
          #first_term = a-l == 1 ? 0.0 : g[a,a-1]*δ[a-l-1]
          #
          #γ[a] = ( first_term + g[a,a]*γ[a-l] + g[a,a+1]*δ[a-l] - δ[a-1]*g[a-1,a] ) / g[a,a]
          γ[a] = ( g[a,a]*γ[a-l] + g[a,a+1]*δ[a-l] - δ[a-1]*g[a-1,a] ) / g[a,a]
          δ[a] = (g[a+1,a+1]*δ[a-l]) / g[a,a]
        end

        println("Result :: (a =  ",  a, " ) ", γ[a], " , ", δ[a], " ")

        third_coeff = (a == 1) ? 0.0 : δ[a-1]

        if a == 1
          Z1[:,a+1] .=@. Z2[:,a+1] + (σ - γ[a])*Z1[:,a]
        else
          Z1[:,a+1] .=@. Z2[:,a+1] + (σ - γ[a])*Z1[:,a] - δ[a-1]*Z1[:,a-1]
        end
        Z1[:,a+1] .=@. Z1[:,a+1] / δ[a]

        for k in 2:l
          Zk   = view(Z, : , ((k-1)*zsize+1):k*zsize)
          Zkp1 = view(Z, : , (k*zsize+1):(k+1)*zsize)

          Zk[:,a+k] .=@. Zkp1[:,a+k] + (σ - γ[a])*Zk[:,a+k-1] - third_coeff*Zk[:,a+k-2]
          Zk[:,a+k] .=@. Zk[:,a+k] / δ[a]
        end

        zlp1ip1 .=@. zlp1ip1 - γ[a]*Zlp1[:,i] - third_coeff*Zlp1[:,i-1]
        zlp1ip1 .=@. zlp1ip1 / δ[a]
      end # a >= 1

      for j in max(a-l+1,1):a+1; g[j,i+1] = dot(Zlp1[:, i+1], Z1[:, j]); end
      for j in max(a+2,1):i+1;   g[j,i+1] = dot(Zlp1[:, i+1], Zlp1[:, j]); end


      if a == 1
        η = γ[1]
        ζ = rnorm
        p .=@. z11 / η
      elseif a >= 2
        λ = δ[a-1] / η
        η = γ[a] - λ*δ[a-1]
        x .=@. x + ζ*p
        ζ = - λ*ζ
        p .=@. (Z1[:, a] - δ[a-1]*p) / η

      end

    end #for iters

    # Compute true residual
    Mat(r, x)

    r .=@. b - r
    rnorm = sqrt(norm2(r))

    param.true_res = rnorm / bnorm
    if verbose == true; println("GMRes: Relative residual: true = ", param.true_res); end

    #println("Hessenberg matrix\n\n", Hm1m)

end # solver

end #QJuliaPLCG2
