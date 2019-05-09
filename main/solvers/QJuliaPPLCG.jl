module QJuliaPPLCG

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

@inline function MatPrecon(out::AbstractArray, inp::AbstractArray, outSloppy::AbstractArray, inpSloppy::AbstractArray, K::Function)

	if pointer_from_objref(out) == pointer_from_objref(inp); return; end #nothing to do

	outSloppy .=@. 0.0
	cpy(inpSloppy, inp)       #noop for the alias refs
	K(outSloppy, inpSloppy)
	cpy(out, outSloppy)       #noop for the alias refs

end


function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function, extra_args...)

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
    data_type = typeof(x[1]) == BigFloat ? BigFloat : Float64

    local r    = zero(typeof(x)(undef, length(x)))
    local p    = zero(typeof(x)(undef, length(x)))
    # Basis vectors
    local Vm1  = zero(Matrix{typeof(x[1])}(undef, length(x), (nkrylov+1+l)))
    local Zm1  = zero(Matrix{typeof(x[1])}(undef, length(x), (nkrylov+1+l)))
    local Um1  = zero(Matrix{typeof(x[1])}(undef, length(x), (nkrylov+1+l)))		

    # G matrix
    local g = zero(Matrix{data_type}(undef, (nkrylov+1+4l), nkrylov+4l))

    local γ = zero(Vector{data_type}(undef, (nkrylov+1)))
    local δ = zero(Vector{data_type}(undef, (nkrylov+1)))

    bnorm = sqrt(norm2(b))  #Save norm of b
    rnorm = 0.0     #if zero source then we will exit immediately doing no work

    v1 = view(Vm1, :, 1)
    z1 = view(Zm1, :, 1)

    if param.use_init_guess == true
      #r = b - Ax0
      Mat(v1, x)
      v1 .=@. v1 - σ*x
      v1 .=@. b - v1
      rnorm = sqrt(norm2(v1))
    else
      rnorm = bnorm
      v1 .=@. b
    end

    if( rnorm == 0.0 ); return; end
    if verbose == true; println("LPCG: Initial residual = ", rnorm); end

    # Scale the first basis vector:
    v1 .=@. (1.0 / rnorm) * v1
    z1 .=@. v1
    #
    g[1,1] = 1.0
    # CG params:
    η = 0.0; ζ = 0.0; λ = 0.0

    #start iterations:
    for i in 1:(min(param.maxiter, nkrylov)+l)
      #
      zip1 = view(Zm1, :, i+1)
      zi   = view(Zm1, :, i)
      #
      MatSloppy(zip1, zi)
      #
      if i <= l; zip1 .=@. zip1 - σ * zi; end # apply shift σ -> σ[i]
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
          println("SQRT breakdown detected.")
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
        println("Result :: ", γ[a], " :: ", δ[a], " ")
        vap1 = view(Vm1, :, a+1)
        zap1 = view(Zm1, :, a+1)
        zim1 = view(Zm1, :, i-1)

        vap1 .=@. zap1
        for j in max(a-2l+1, 1):a
          vj = view(Vm1,:,j)
          vap1 .=@. vap1 - g[j,a+1]*vj
        end
        vap1 .=@. vap1 / g[a+1,a+1]

        third_coeff = a == 1 ? 0.0 : δ[a-1]

        zip1 .=@. zip1 - γ[a]*zi - third_coeff*zim1
        zip1 .=@. zip1 / δ[a]
      end # a >= 1

      if a < 1
        for j in 1:i+1; g[j,i+1] = dot(Zm1[:, i+1], Zm1[:, j]); end
      else
        for j in max(a-l+1,1):a+1; g[j,i+1] = dot(Zm1[:, i+1], Vm1[:, j]); end
        for j in max(a+2,1):i+1;   g[j,i+1] = dot(Zm1[:, i+1], Zm1[:, j]); end
      end

      if a == 1
        η = γ[1]
        ζ = rnorm
        p .=@. v1 / η
      elseif a >= 2
        λ = δ[a-1] / η
        η = γ[a] - λ*δ[a-1]
        x .=@. x + ζ*p
        ζ = - λ*ζ
        p .=@. (Vm1[:, a] - δ[a-1]*p) / η

      end

    end #for iters

    # Compute true residual
    Mat(r, x)

    r .=@. b - r
    rnorm = sqrt(norm2(r))

    param.true_res = rnorm / bnorm
    if verbose == true; println("GMRes: Relative residual: true = ", param.true_res); end

    #println("Hessenberg matrix\n\n", Hm1m)

end #MR

end #QJuliaLanMR
