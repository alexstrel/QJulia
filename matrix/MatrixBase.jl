module MatrixBase

using MatrixMarket
using Base.Threads

ex = :(macro dowhile(branch, sym, cond)
           quote
               s = string($sym)
               $sym != :while && error("expected :while symbol, got :$s")
               $(esc(branch))
               while $(esc(cond))
                   $(esc(branch))
               end
           end
       end)

ex.args[1].args[1] = :do
@eval $ex

abstract type GenericCSRMat end

mutable struct CSRMat{T<:Any} <: GenericCSRMat

  # Is complex
  is_complex::Bool

  # nrows, ncols
  Dims::NTuple{2, Int}

  # Iterators
  X::Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}

  # Data arrays
  csrVals::Vector{T}
  csrRows::Vector{Int32}
  csrCols::Vector{Int32}
  diagIdx::Vector{Int32} #for ILU etc.

  # Avg number of nz elements in a row
  N::Int

  function CSRMat{T}(path::String) where T

    is_complex = (T == ComplexF16 || T == ComplexF32 || T == ComplexF64)

    M = MatrixMarket.mmread(path)

    #collect basic info
    (nrows, ncols) = size(M)
    if nrows != ncols; error("Rectangle matrices are not allowed"); end

    (rows, cols) = axes(M)

    prec = eltype(M)

    nnz = 0
    for i in M
      if i != 0.0; nnz += 1; end
    end

    println("Element type : ", prec,", number of nnz : ", nnz, ", number of rows : ", nrows, ", number of cols : ", ncols)

    #convert into CSR format.
    csrVals = Vector{prec}(undef, nnz)
    csrRows = Vector{Int32}(undef, nrows+1)
    csrCols = Vector{Int32}(undef, nnz)
    diagIdx = Vector{Int32}(undef, nrows)

    #construct CSR matrix (with fortran indexing convention)
    csrRows[1] = 1

    e = 1
    c = 1
    d = 1

    avg_row_nnz = 0
    #
    for j in rows
      row_nnz = 0
      for i in cols
        if M[j,i] != 0.0

          csrVals[e] = convert(T, M[j,i])
          csrCols[c] = i
          if j == i
            diagIdx[d] = i; d +=1
          end #store diag index

          row_nnz += 1; e += 1; c += 1

        end #if
      end #for i
      csrRows[j+1] = csrRows[j] + row_nnz

      avg_row_nnz += row_nnz
    end #for j

    if csrRows[nrows+1] != nnz+1; error("Conversion failed, incorrect number of nnz detected."); end

    avg_row_nnz = round( Float64(avg_row_nnz) / Float64(nrows))
    # call constructor
    new(is_complex, (nrows, ncols), (rows, cols), csrVals, csrRows, csrCols, view(diagIdx, 1:(d-1)), avg_row_nnz)

  end #CSRMat{T}

end #CSRMat


function print_CSRMat_info(m::GenericCSRMat)
  println(" ")
  println("General info for ", typeof(m), ": ")
  println("Element type : ", typeof(m.csrVals[1]),", number of nnz : ", (m.csrRows[m.Dims[1]+1] - 1) , ", number of rows : ", m.Dims[1], ", number of cols : ", m.Dims[2], " average number of nz elements in a row ", m.N)
end

function csrmv(b::AbstractArray, m::GenericCSRMat, x::AbstractArray)

@threads  for j in m.X[1]
#  for j in m.X[1]
    offset  = m.csrRows[j] - m.csrRows[1]
    row_nnz = m.csrRows[j+1] - m.csrRows[j]
    b[j]    = 0.0

    for i in 1:row_nnz
      cid   = m.csrCols[offset+i]
      b[j] += m.csrVals[offset+i]*x[cid]
    end

  end

end

function csrmv(m::GenericCSRMat, x::AbstractArray)

   b = zero(typeof(x)(undef, length(x)))
@threads  for j in m.X[1]
#  for j in m.X[1]
    offset  = m.csrRows[j] - m.csrRows[1]
    row_nnz = m.csrRows[j+1] - m.csrRows[j]
    b[j]    = 0.0

    for i in 1:row_nnz
      cid   = m.csrCols[offset+i]
      b[j] += m.csrVals[offset+i]*x[cid]
    end

  end

  return b
end


function ilu0(M::GenericCSRMat)

  println("Perform ILU decomposition on matrix ", typeof(M) )

  m = deepcopy(M)

  iw  = Vector{Int}(undef, m.Dims[1])

  for i in m.X[1]
    iw .=@. -1 #elementwise assingment

    for k in m.csrRows[i]:(m.csrRows[i+1] - 1); iw[m.csrCols[k]] = k; end

    j = m.csrRows[i]

    @do begin

      jrow = m.csrRows[j]
      if i <= jrow; break; end

      m.csrVals[j] *= m.csrVals[m.diagIdx[jrow]]

      tl = m.csrVals[j]

      for jj in (m.diagIdx[jrow] + 1):(ia[jrow+1] - 1)

        jw = iw[m.csrCols[jj]]
        if jw != -1; l[jw] -= (tl * l[jj]); end

      end #for jj
      j += 1

    end :while j <= (m.csrRows[i+1] - 1)

    m.diagIdx[i] = j

    if jrow != i
      error("ILU_CR - Fatal error! JROW != I, JROW = ", jrow,", I = ", i)
    end

    if  m.csrVals[j] == 0.0
      error("ILU_CR - Fatal error! Zero pivot on step I = ", i,", L[j] = 0.0 for j = ", j)
    end

    m.csrVals[j] = 1.0 / m.csrVals[j];
  end #for i

  for k in m.X[1]; m.csrVals[m.diagIdx[k]] = 1.0 / m.csrVals[m.diagIdx[k]]; end

  return m

end #ilu0

function ilu0(out::AbstractArray, m::GenericCSRMat, inp::AbstractArray)

  out .=@. inp

#  Solve L * w = w where L is unit lower triangular.

  for i in 2:length(tmp)
    for j in m.csrRows[i]:(m.diagIdx[i]-1); out[i] -= m.csrVals[i]*out[m.csrCols[j]]; end
  end

#  Solve U * w = w, where U is upper triangular.

  for i in length(tmp):-1:1
    for j in (m.diagIdx[i] + 1):m.csrRows[i+1]; out[i] -= m.csrVals[j]*out[m.csrCols[j]]; end
    out[i] /= m.csrVals[m.diagIdx[i]]
  end
end #ilu0


end #MatrixBase
