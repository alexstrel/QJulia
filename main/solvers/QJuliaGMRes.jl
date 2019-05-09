module QJuliaGMRes

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaReduce
using QJuliaSolvers

using LinearAlgebra
using MPI
using Printf

norm2   = QJuliaReduce.gnorm2
rdot    = QJuliaReduce.reDotProduct
verbose = true

const ε = eps(Float64)

# j - current row index , k - current column index

@inline function HessenbergQR(Givens::Any, H::Any, j, k; compute_last_element = false)
    #extract component
    cs   = Givens[1]
    sn   = Givens[2]
    Rm1m = Givens[3]
    γ    = Givens[4]

    if compute_last_element == true # last entry in the column (all columns), that means j = k
      if (j != k) error("Wrong row index. Must be diagonal element."); end
      inv_denom = 1.0 / sqrt(sn[k]*sn[k]+H[j+1,j]*H[j+1,j])
      old_snj   = sn[k]
      cs[j]     = old_snj * inv_denom; sn[j] = H[j+1,j] * inv_denom
      Rm1m[j,j] = cs[j]*old_snj + sn[j]*H[j+1,j]
      return
    end

    if j == 1 # initialization stage for all columns
      sn[k] = H[j,j] # use sn[j] is a temp variable
      return
    end

    Rm1m[j-1, k] = cs[j-1]*sn[k] + sn[j-1]*H[j,k]
    sn[k] = -sn[j-1]*sn[k] + cs[j-1]*H[j,k]

end

@inline function PlainArnoldiCycle(Mat::Any, Givens::Any, H::Any, Vk1::Any, k, param; do_givens_qr = true)

    k1 = k+1
    σ  = param.shift

    if k < 1; error("Wrong vector dimension in the Arnodli step."); end

    w = view(Vk1, :, k1)
    v = view(Vk1, :, k )

    Mat(w, v)

    # Apply shifts:
    w .=@. w - σ*v

    for j in 1:k
      H[j,k] = rdot(Vk1[:,k1], Vk1[:,j])
      Vk1[:,k1] .=@. Vk1[:,k1] - H[j,k] * Vk1[:,j]

      if do_givens_qr == true; HessenbergQR(Givens, H, j, k); end
    end

    H[k1,k] = norm(Vk1[:,k1])
    Vk1[:,k1] .=@. 1.0 / H[k1,k] * Vk1[:,k1]

    if do_givens_qr == true; HessenbergQR(Givens, H, k, k; compute_last_element = true); end

    return
end

@inline function PipelinedArnoldiCycle_l1(Mat::Any, Givens::Any, H::Any, Vk1::Any, k, param; do_givens_qr = false)

    k1 = k+1
    σ  = param.shift

    if k < 1; error("Wrong vector dimension in the Arnodli step."); end

    v = view(Vk1, :, k )
    u = view(Vk1, :, k1)
    z = zeros(typeof(Vk1[1,k]), length(Vk1[:,k]))

    Mat(z, v)

    # Apply shifts:
    z .=@. z - σ*v

    # Prepare subdiag. element of Hess. matrix and the last Arnoldi vector:
    H[k1, k] = norm2(z)
    u .=@. z

    for j in 1:k
      vj        = view(Vk1, :, j)
      # WARNING: vj is a subarray so rdot z, vj won't work
      H[j,k]    = dot(z, vj)

      H[k1, k] -= H[j,k]*H[j,k]
      u .=@. u - H[j,k]*vj

      if j == k; H[j,k] += σ; end
      if do_givens_qr == true; HessenbergQR(Givens, H, j, k); end
    end

    if H[k1, k] <= 0.0; error("SQRT breakdown detected at Arnoldi cycle ", k, " ."); end

    H[k1,k] = sqrt(H[k1,k])
    u .=@. u / H[k1,k]

    return
end


@inline function PipelinedArnoldiCycle_l1_step1(Mat::Any, Givens::Any, H::Any, Vk::Any, k, param; do_givens_qr = false)

    k1      = k+1
    σ       = param.shift
    nkrylov = param.nKrylov

    if k < 1; error("Wrong vector dimension in the Arnodli step."); end

    Zk1 = view(Vk, :, nkrylov+2:2nkrylov+3)

    if k == 1; Zk1[:, k] .=@. Vk[:, k]; end

    z = view(Zk1, :, k )
    u = zeros(typeof(Vk[1,k]), length(Vk[:,k]))
    w = zeros(typeof(Zk1[1,k]), length(Zk1[:,k]))

    Mat(w, z)

    znorm2 = norm2(z)

    if k == 1 #compute intital values
      znorm = sqrt(znorm2)
      Vk[:,k]   .=@. 1.0 / znorm * Vk[:,k]
      Zk1[:,k1] .=@. 1.0 / znorm * w
      return;
    end

    # Prepare subdiag. element of Hess. matrix and the last Arnoldi vector:
    H[k, k-1] = znorm2
    u .=@. z

    for j in 1:(k-1)
      vj   = view(Vk, :, j  )
      zj1  = view(Zk1,:, j+1)
      # WARNING: vj is a subarray so rdot z, vj won't work
      H[j,k-1] = dot(z, vj)

      H[k, k-1] -= H[j,k-1]*H[j,k-1]
      u .=@. u - H[j,k-1]*vj
      w .=@. w - H[j,k-1]*zj1

      if j == k; H[j,k-1] += σ; end
      if do_givens_qr == true; HessenbergQR(Givens, H, j, k); end
    end

    if H[k, k-1] <= 0.0; error("SQRT breakdown detected at Arnoldi cycle ", k, " ."); end

    H[k,k-1] = sqrt(H[k,k-1])

    Vk[:,k]   .=@. u / H[k,k-1]
    Zk1[:,k1] .=@. w / H[k,k-1]

    return
end


@inline function PipelinedArnoldiCycle_p1(Mat::Any, Givens::Any, H::Any, Vk::Any, k, param; do_givens_qr = false)

    σ       = param.shift
    nkrylov = param.nKrylov

    if k < 1; error("Wrong vector dimension in the Arnodli step."); end
    if do_givens_qr == true; error("Givens QR is not supported."); end

    Zkp1 = view(Vk, :, nkrylov+2:2nkrylov+3)

    if k == 1; Zkp1[:, k] .=@. Vk[:, k]; end
    if k == (nkrylov + 2)
      Vk[:, k-1] .=@. Vk[:, k-1] / H[k-1, k-2]
      return;
    end

    zk   = view(Zkp1, :, k  )
    zkp1 = view(Zkp1, :, k+1)
    u = zeros(typeof(Vk[1,k]), length(Vk[:,k]))

    Mat(zkp1, zk)

    if k > 2
      vkm1  = view(Vk, :, k-1)
      vkm1 .=@. vkm1 / H[k-1,k-2]
      zk   .=@. zk   / H[k-1,k-2]
      zkp1 .=@. zkp1 / H[k-1,k-2]

      for j in 1:(k-2); H[j,k-1] /= H[k-1,k-2]; end

      H[k-1,k-1] /= (H[k-1,k-2]*H[k-1,k-2])
      H[k-1,k-1] += σ #must be sigma[k-1] !
    end

    if k > 1
      vk = view(Vk, :, k)
      vk .=@. zk
      for j in 1:(k-1)
        vj    = view(Vk, :, j)
        vk   .=@. vk - H[j,k-1]*vj

        if k > nkrylov; continue; end  # in this case we don't need the next Z-vector

        zjp1  = view(Zkp1, :, j+1)
        zkp1 .=@. zkp1 -  H[j,k-1]*zjp1
      end
      H[k, k-1] = norm(vk)
    end

    if k > nkrylov; return; end

    for j in 1:k
      vj   = view(Vk, :, j)
      # WARNING: vj is a subarray so rdot z, vj won't work
      H[j,k] = dot(zkp1,vj)
    end

    return
end



const do_givens = false
#TODO: givens version is broken
const pipeline_algo_type = 3

function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj)

    if verbose == true; println("Running GMRES in maximum " , param.nKrylov, " iters."); end

    QJuliaReduce.set_blas_global_reduction(param.global_reduction)

    if MPI.Initialized() == false; error("MPI was not inititalized, copy source field to the solution."); end

    if param.nKrylov > param.maxiter; println("Warning: search space exceeds requested maximum iter count. Some routines may fail."); end

    if (param.maxiter == 0) || (param.nKrylov == 0)
      if param.use_init_guess == false
        x .=@. 0.0
      end
      return
    end #if

    mixed = (param.dtype_sloppy != param.dtype)

    σ = param.shift
    nkrylov = param.nKrylov
    data_type = typeof(x[1]) == BigFloat ? BigFloat : Float64

    local r    = zero(typeof(x)(undef, length(x)))
    # Arnoldi vectors
    set_size    = pipeline_algo_type < 2 ? nkrylov+1 : (nkrylov+1) + (nkrylov+2)
    local Vm1   = zero(Matrix{typeof(x[1])}(undef, length(x), set_size))

    if do_givens == true
      # Givens objects:
      # R matrix from givens QR of H (use triu() for explicit upper triangular view or tril for lower triangular view)
      local Rm1m  = zero(Matrix{BigFloat}(undef, (nkrylov+1), nkrylov))
      # Givens matrix coeffs
      local cs    = zeros(BigFloat, nkrylov)
      local sn    = zeros(BigFloat, nkrylov)
    end

    # Hessenberg matrix
    local Hm1m  = zero(Matrix{data_type}(undef, (nkrylov+1), nkrylov))

    bnorm = sqrt(norm2(b))  #Save norm of b
    rnorm = 0.0     #if zero source then we will exit immediately doing no work

    v0 = view(Vm1, :, 1)

    if param.use_init_guess == true
      #r = b - Ax0
      Mat(v0, x)
      v0 .=@. v0 - σ*x
      v0 .=@. b - v0
      rnorm = sqrt(norm2(v0))
    else
      rnorm = bnorm
      v0 .=@. b
    end

    if( rnorm == 0.0 ); return; end
    if verbose == true; println("GMRES: Initial residual = ", rnorm); end

    # Scale the first basis vector:
    if pipeline_algo_type != 2;  v0 .=@. (1.0 / rnorm) * v0; end
    # Aux coeffs
    β = rnorm

    # LS problem source vector
    γ  = zeros(BigFloat, nkrylov+1); γ[1] = β

    # Single object for all components
    Gm1m =  do_givens == true ? (cs, sn, Rm1m, γ) : nothing

    #start iterations
    ss = min(param.maxiter, param.nKrylov)

    if pipeline_algo_type == 0
      println("Running plain Arnoldi cycles.")
      ArnoldiCycle = PlainArnoldiCycle
    elseif pipeline_algo_type == 1
      println("Running pipelined Arnoldi cycles.")
      ArnoldiCycle = PipelinedArnoldiCycle_l1
    elseif pipeline_algo_type == 2
      println("Running pipelined Arnoldi cycles (s1 version).")
      ArnoldiCycle = PipelinedArnoldiCycle_l1_step1
      ss += 1   #+1 iteration for this type
    elseif pipeline_algo_type == 3
      println("Running pipelined Arnoldi cycles (p1 version).")
      ArnoldiCycle = PipelinedArnoldiCycle_p1
      ss += 2   #+2 iteration for this type
    else
      error("Arnoldi step is not selected.")
    end

    for k in 1:ss
      # Arnoldi step + Givens QR:
      ArnoldiCycle(MatSloppy, Gm1m, Hm1m, Vm1, k, param; do_givens_qr = do_givens)
      # Update LS source vector and compute iter residual
      if do_givens == true
        γ[k+1] = -sn[k]*γ[k]
        γ[k]   = γ[k]*cs[k]
        rnorm  = abs(γ[k+1])
      end

      if(verbose == true && do_givens == true); println("GMRes: ", k,"  residual norm = ", rnorm); end
    end #for itern
    # compute solution vector:
    if do_givens == false
      # native QR method (access via .Q and .R )
      qrH = qr(Hm1m)
      Rmm = qrH.R
      Qγ  = transpose(qrH.Q) * γ
      γ_  = view(Qγ,1:nkrylov)
    else
      Rmm = triu(view(Rm1m, 1:nkrylov, :))
      γ_  = view(γ,1:nkrylov)
    end
    η   = Rmm \ γ_

    for i in 1:nkrylov
      x .=@. x + Vm1[:, i]*η[i]
    end

    # Compute true residual
    Mat(r, x)

    r .=@. b - r
    rnorm = sqrt(norm2(r))

    param.true_res = rnorm / bnorm
    if verbose == true; println("GMRes: Relative residual: true = ", param.true_res); end

    println("Hessenberg matrix\n\n")
    for ii in 1:(nkrylov+1)
      for jj in 1:2
        @printf(": %1.10e ",Hm1m[ii,jj] )
      end
      println("\n------------------------------\n")
    end

    QJuliaReduce.reset_blas_global_reduction()

end #MR

end #QJuliaLanMR
