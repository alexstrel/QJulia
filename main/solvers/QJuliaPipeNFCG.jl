module QJuliaPipeNFCG

using QJuliaInterface
using QJuliaEnums
using QJuliaBlas
using QJuliaReduce
using QJuliaSolvers

using LinearAlgebra
using Printf

using Plots

norm2    = QJuliaReduce.gnorm2
rdot     = QJuliaReduce.reDotProduct
cpy      = QJuliaBlas.cpy


@enum NormType begin
	  preconditioned_norm
	  unpreconditioned_norm
	  natural_norm
  end

@enum TruncType begin
      trunc_type_standard # <- bad with the preconditioner
	  trunc_type_notay    # <- significantly better with a preconditioner
	end


@inline function MatPrecon(out::AbstractArray, inp::AbstractArray, outSloppy::AbstractArray, inpSloppy::AbstractArray, K::Function)

    if pointer_from_objref(out) == pointer_from_objref(inp); return; end #nothing to do

    outSloppy .=@. 0.0
    cpy(inpSloppy, inp)       #noop for the alias refs
    K(outSloppy, inpSloppy)
    cpy(out, outSloppy)       #noop for the alias refs

  end

  function solver(x::AbstractArray, b::AbstractArray, Mat::Any, MatSloppy::Any, param::QJuliaSolvers.QJuliaSolverParam_qj, K::Function, extra_args...)

    is_preconditioned = param.inv_type_precondition != QJuliaEnums.QJULIA_INVALID_INVERTER

    solver_name = "PipeNFCG2"

    normcheck  = unpreconditioned_norm
    truncstrat = trunc_type_notay #trunc_type_standard

    mmax = param.nKrylov

    println("Running ", solver_name ," solver with truncation length ", mmax)

    if is_preconditioned == true; println("Preconditioner  :: ", param.inv_type_precondition); end

    if (param.maxiter == 0)
      if param.use_init_guess == false; x .=@. 0.0; end
      return
    end #if param.maxiter == 0

    mixed = (param.dtype_sloppy != param.dtype)

    if mixed == true; error("Running mixed precision solver.");end

    local r = zeros(param.dtype_sloppy, length(x))
    local u = zeros(param.dtype_sloppy, length(x))
    local w = zeros(param.dtype_sloppy, length(x))
    local m = zeros(param.dtype_sloppy, length(x))
    local n = zeros(param.dtype_sloppy, length(x))
    # set of vectors
    local p  = Matrix{param.dtype_sloppy}(undef, length(x), mmax+1)
    local s  = Matrix{param.dtype_sloppy}(undef, length(x), mmax+1)
    local q  = Matrix{param.dtype_sloppy}(undef, length(x), mmax+1)
    local z  = Matrix{param.dtype_sloppy}(undef, length(x), mmax+1)

    p .=@. 0.0
    s .=@. 0.0
    q .=@. 0.0
    z .=@. 0.0
    #
    local beta  = zeros(Float64, mmax+1)
    local eta   = zeros(Float64, mmax+1)

    dp = 0.0; norm2B = norm(b)

    Precond(out, inp) = MatPrecon(out, inp, out, inp, K)

    # Compute cycle initial residual
    if param.use_init_guess == true
      Mat(r, x)
      r .=@. b - r
    else
      r .=@. b
    end
    # Initial stage
    Precond(u, r)
    Mat(w, u)
    #
    gamma   = dot(r, u)
    delta   = dot(w, u)
    eta[1]  = delta; alpha = gamma/delta
    #
    dp  = normcheck == preconditioned_norm || normcheck == unpreconditioned_norm ? norm(u) : abs(gamma)

    # m = B(w)
    Precond(m, w)
    # n = Am
    Mat(n, m)
    p[:,1] .=@. u
    s[:,1] .=@. w
    q[:,1] .=@. m
    z[:,1] .=@. n
    # update x, r, z, w as zero iteration
    x .=@. x + alpha*p[:,1]
    r .=@. r - alpha*s[:,1]
    u .=@. u - alpha*q[:,1]
    w .=@. w - alpha*z[:,1]
    #
    stop = dp*dp*param.tol*param.tol

    println(solver_name," : Initial residual = ", dp / norm2B)

    gamma   = dot(r, u)
    delta   = dot(w, u)
    beta[1] = dot(s[:,1], u)

    k = 1; converged = false;
    mi = truncstrat == trunc_type_notay ? 1 : mmax

    while (k < param.maxiter && converged == false)

      idx = k % (mmax + 1) + 1
      #m = u + B(w-r)
      n .=@. w - r
      Precond(m, n)
      m .=@. u + m
      #n = Am
      Mat(n, m)

      # finish all global comms here
      eta[idx] = 0.0 
      j = 0
      for i in max(0,k-mi):(k-1)
        kdx = (i % (mmax+1)) + 1; j += 1
	beta[j] /= -eta[kdx]
	eta[idx] -= ((beta[j])*(beta[j])) * eta[kdx]
      end
      # 
      eta[idx] += delta
      if(eta[idx] <= 0.)
        println("Restart due to square root breakdown or exact zero of eta at it = ", k)
        break
      else
        alpha = gamma/eta[idx]
      end

      # project out stored search directions
      p[:,idx] .=@. u
      s[:,idx] .=@. w
      q[:,idx] .=@. m
      z[:,idx] .=@. n

      j = 0
      for i in max(0,k-mi):(k-1)
        kdx = (i % (mmax+1)) + 1; j += 1
	#
	p[:,idx] .=@. p[:,idx] + beta[j] * p[:,kdx]
	s[:,idx] .=@. s[:,idx] + beta[j] * s[:,kdx]
	q[:,idx] .=@. q[:,idx] + beta[j] * q[:,kdx]
	z[:,idx] .=@. z[:,idx] + beta[j] * z[:,kdx]
      end

      # Update x, r, z, w
      x .=@. x + alpha*p[:,idx]
      r .=@. r - alpha*s[:,idx]
      u .=@. u - alpha*q[:,idx]
      w .=@. w - alpha*z[:,idx]

      mi = truncstrat == trunc_type_notay ? ((k) % mmax)+1 : mmax

      gamma = dot(r, u)
      delta = dot(w, u)
      j = 0
      for i in max(0,k-mi+1):k
        kdx = (i % (mmax+1)) + 1; j += 1
	beta[j] = dot(s[:,kdx], u)
      end

      dp = normcheck == preconditioned_norm ? norm(u) : (normcheck == unpreconditioned_norm ? norm(r) : sqrt(gamma))
      # Check for convergence
      @printf("%s: %d iteration, iter residual: %1.15e \n", solver_name, k, dp/norm2B)

      k += 1
    end # while context

    return
  end # solver context

end # module cobtext
