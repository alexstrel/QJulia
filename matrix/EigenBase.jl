module EigenBase

@enum Eigen4JuliaWrapperPrecondType begin
  EIGENBASE_PRECOND_NONE
  EIGENBASE_PRECOND_ILU
  EIGENBASE_PRECOND_ICC
end

@enum Eigen4JuliaWrapperPrecisionType begin
  EIGENBASE_FLOAT_SINGLE = 4
  EIGENBASE_FLOAT_DOUBLE = 8
end

@enum Eigen4JuliaWrapperCompressionType begin
  EIGENBASE_CRS_FORMAT = 1
  EIGENBASE_CCS_FORMAT = 2
end

mutable struct PrecondDescr
  # preconditioner handle
  pc_args::Ptr{Cvoid}
  # preconditioner type
  pctype::Eigen4JuliaWrapperPrecondType
  # problem matrix dimensions
  rows::Cint
  cols::Cint
  # precision of the preconditioner
  precision::Eigen4JuliaWrapperPrecisionType
  # complex flag
  is_complex::Cint
  # sparse format
  ctype::Eigen4JuliaWrapperCompressionType
  # nnz of the original matrix
  nnz::Cint
  # initialization flag
  is_init::Cint

  PrecondDescr() = new(C_NULL, EIGENBASE_PRECOND_NONE, 0,0,EIGENBASE_FLOAT_DOUBLE,0,EIGENBASE_CRS_FORMAT,0,0)

end #PrecondDescr

CreateEigenPreconditioner_qj(vals,innerIndices,outerIndexPtr,descr) = ccall((:CreatePreconditioner, "libeigenpc"), Cvoid, (Ptr{Cvoid},Ptr{Cint},Ptr{Cint},Ref{PrecondDescr}),vals,innerIndices,outerIndexPtr,descr)

ApplyEigenPreconditioner_qj(out, in, descr) = ccall((:ApplyPreconditioner, "libeigenpc"), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ref{PrecondDescr}),out, in, descr)

DestroyEigenPreconditioner_qj(descr) = ccall((:DestroyPreconditioner, "libeigenpc"), Cvoid, (Ref{PrecondDescr},),descr)

end #EigenBase
