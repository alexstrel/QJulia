module QJuliaLanMR

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaReduce
using QJuliaSolvers

using LinearAlgebra
using MPI

norm2   = QJuliaReduce.gnorm2
rdot    = QJuliaReduce.reDotProduct
verbose = false

const ε = eps(Float64)

@inline function LanczosStep(Mat::Any, vk::AbstractArray, vkm1::AbstractArray, βold::Float64, σ::Float64, pk::AbstractArray)
    Mat(pk, vk)
    pk .=@. pk - βold*vkm1
    # Compute Lancsoz coefficient α
    α = rdot(vk, pk)
    pk .=@. pk - (α+σ)*vk
    # Compute Lancsoz coefficient β
    β = sqrt(rdot(pk, pk))
    if β  < ε ; error("Algorithm breakdown."); end
    vkm1 .=@. vk
    vk .=@. (1.0 / β)*pk
    return  α, β
end

function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj)

    if verbose == true; println("Running Lanczos version of MR solver in maximum " , param.Nsteps, " step(s)."); end

    QJuliaReduce.set_blas_global_reduction(param.global_reduction)

    if MPI.Initialized() == false; error("MPI was not inititalized, copy source field to the solution."); end

    if (param.maxiter == 0) || (param.Nsteps == 0)
      if param.use_init_guess == false
        x .=@. 0.0
      end
      return
    end #if

    mixed = (param.dtype_sloppy != param.dtype)

    local r     = zero(typeof(x)(undef, length(x)))
    local Av    = zero(typeof(x)(undef, length(x)))
    # explicit Lanczos vectors
    local vk    = zero(typeof(x)(undef, length(x)))
    local vkm1  = zero(typeof(x)(undef, length(x)))
    # explicit Lanczos matrix
    # this is just (sub-)diagonal elements (but transposed when stored!):
    # #(0) β2      ... βk-1  βk
    # α1   α2   α3 ... αk-1  αk
    # β2   β3   β4 ... βk    βk+1
    #
    # R-matrix has the following structure
    # γ1 δ2 ϵ3  0  0  0  ...
    # 0  γ2 δ3 ϵ4  0  0  ...
    # 0  0  γ3 δ4  ϵ5 0  ...
    # 0 ................ γ_k


    local Tk    = zero(Matrix{Float64}(undef, (param.maxiter+1), 2))
    # some aux fields:
    local dk    = zero(typeof(x)(undef, length(x)))
    local dkm1  = zero(typeof(x)(undef, length(x)))

    bnorm = sqrt(norm2(b))  #Save norm of b
    rnorm = 0.0     #if zero source then we will exit immediately doing no work

    σ = param.shift

    if param.use_init_guess == true
      #r = b - Ax0
      Mat(r, x)
      r .=@. r - σ*x
      r .=@. b - r
      rnorm = sqrt(norm2(r))
    else
      rnorm = bnorm
      r .=@. b
    end

    if( rnorm == 0.0 ); return; end
    if verbose == true; println("MR: Initial residual = ", rnorm); end

    # Load init vectors
    vk   .=@. (1.0 / rnorm) * r
    # Givens parameters
    cs = -1.0; sn = 0.0
    # Matrix norm:
    Anorm = 0.0

    # Lancsoz coeffs
    α = 0.0; β = 0.0; βold = rnorm
    # aux parameters
    δk   = 0.0; ϵold = 0.0;
    #start iterations
    for k in 1:param.maxiter
      # Lanczos step:
      α, β = LanczosStep(Mat, vk, vkm1, βold, σ, Av); Tk[k,1] = α; Tk[k,2] = β
      # Givens QR
      # produce middle two enties in the last coloumn of Tk
      δ  = cs*δk + sn*α
      γk = sn*δk - cs*α
      # produce first two entries of T_{k+1}*e_{k+1}
      ϵ  =  sn*β
      δk = -cs*β
      # eliminate β
      γinv  = 1.0 / sqrt(γk*γk+β*β)
      sn    = β  * γinv
      cs    = γk * γinv
      # Solution update (using Av as tmp)
      Av   .=@. γinv*vkm1 - (γinv*δ)*dk - (γinv*ϵold)*dkm1
      dkm1 .=@. dk
      dk   .=@. Av
      #
      τk = rnorm*cs
      x .=@. x + τk*dk
      #residual norm and matrix norm:
      Arnorm = rnorm*sqrt(γk*γk+δk*δk)
      #
      rnorm  *= sn
      #
      Anorm = k == 1 ? sqrt(α*α+β*β) : max(Anorm, sqrt(βold*βold+α*α+β*β))

      if(verbose == true); println("Lanczos MR: ", k,"  residual norm = ", rnorm, "\t ( estimated matrix norm is ", Anorm,  ")"); end
      # Store current β and ϵ for the next cycle
      βold = β; ϵold = ϵ

    end #for itern

    Mat(r, x)

    r .=@. b - r
    rnorm = sqrt(norm2(r))

    param.true_res = rnorm / bnorm
    if verbose == true; println("MR: Relative residual: true = ", param.true_res); end

    QJuliaReduce.reset_blas_global_reduction()

end #MR

end #QJuliaLanMR
