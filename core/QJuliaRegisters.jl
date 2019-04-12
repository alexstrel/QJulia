module QJuliaRegisters

# Plain baseline registers
const half2 = NTuple{2, VecElement{Float16}}
const half4 = NTuple{4, VecElement{Float16}}
#
const float2  = NTuple{2,  VecElement{Float32}}
const float4  = NTuple{4,  VecElement{Float32}}
const float8  = NTuple{8,  VecElement{Float32}}
const float16 = NTuple{16, VecElement{Float32}}
#
const double2  = NTuple{2,  VecElement{Float64}}
const double4  = NTuple{4,  VecElement{Float64}}
const double8  = NTuple{8,  VecElement{Float64}}

# Helper functions for plain registers

function register_type(T)
  if     T == Float16 || T == ComplexF16 || T == half2 || T == half4
    return Float16
  elseif T == Float32 || T == ComplexF32 || T == float2 || T == float4 || T == float8 || T == float16
    return Float32
  elseif T == Float64 || T == ComplexF64 || T == double2 || T == double4 || T == double8
    return Float64
  elseif T == BigFloat || T == Complex{BigFloat}
    return BigFloat
  else
    error("Cannot deduce a type.")
  end

  return nothing
end


function register_size(T)

  if     T == Float16 || T == Float32 || T == Float64 || T == BigFloat
    return 1
  elseif T == ComplexF16 || T == ComplexF32 || T == ComplexF64 || T == Complex{BigFloat} || T == half2 || T == float2 || T == double2
    return 2
  elseif T == double4 || T == float4 || T == half4
    return 4
  elseif T == double8 || T == float8
    return 8
  elseif T == float16
    return 16
  else
    error("Cannot deduce the register size (type is not supported).")
  end

  return nothing
end


# More generic registers

abstract type GenericRegister{T<:Real,N} end

mutable struct FloatN{T<:AbstractFloat,N} <: GenericRegister{T,N}

  val::NTuple{N,  VecElement{T}}

  FloatN{T,N}() where {T} where {N} = new( NTuple{N,  VecElement{T}}(ntuple(i->0, N)) )
  FloatN{T,N}(src::FloatN) where {T} where {N}  = new( src.val  )
  FloatN{T,N}(reg::NTuple{N, VecElement{T}}) where {T} where {N}  = new( reg )

end

# re-implement plain complex type for compatibility

mutable struct Complex2{T<:AbstractFloat, N} <: GenericRegister{T,N}

  val::Complex{T}

  Complex2{T,N}() where {T} where {N} = new( Complex{T}(0.0, 0.0) )
  Complex2{T,N}(src::Complex2) where {T} where {N} = new( src.val  )
  Complex2{T,N}(reg::Complex{T}) where {T} where {N} = new( reg )
  Complex2{T,N}(rea::T, img::T) where {T} where {N} = new( Complex{T}(rea, img) )
  Complex2{T,N}(tpl::NTuple{2, VecElement{T}}) where {T} where {N} = new( Complex{T}(tpl[1], tpl[2]) )

end

mutable struct IntN{T<:Integer,N} <: GenericRegister{T,N}

  val::NTuple{N,  VecElement{T}}

  IntN{T,N}() where {T} where {N} = new( NTuple{N,  VecElement{T}}(ntuple(i->0, N)) )
  IntN{T,N}(src::IntN) where {T} where {N}  = new( src.val  )
  IntN{T,N}(reg::NTuple{N, VecElement{T}}) where {T} where {N}  = new( reg )

end

# Floating point registers
Half2    = FloatN{Float16, 2}
Half4    = FloatN{Float16, 4}
Half8    = FloatN{Float16, 8}
Half16   = FloatN{Float16, 16}
Single2  = FloatN{Float32, 2}
Single4  = FloatN{Float32, 4}
Single8  = FloatN{Float32, 8}
Single16 = FloatN{Float32, 16}
Double2  = FloatN{Float64, 2}
Double4  = FloatN{Float64, 4}
Double8  = FloatN{Float64, 8}
# Big float type
BigDouble2  = FloatN{BigFloat, 2}

# Additional complex types (are they compatible with Half2, Single2, Double2?)
ComplexH = Complex2{Float16,2}
ComplexS = Complex2{Float32,2}
ComplexD = Complex2{Float64,2}
ComplexB = Complex2{BigFloat,2}

# Signed: BigInt Int128 Int16 Int32 Int64 Int8
# Unsigned: UInt128 UInt16 UInt32 UInt64 UInt8

Int2     = IntN{Int32, 2}
Int4     = IntN{Int32, 4}
UInt2    = IntN{UInt32, 2}
UInt4    = IntN{UInt32, 4}
LongInt2 = IntN{Int64, 2}
LongInt4 = IntN{Int64, 4}


# Helper methods
function register_type(reg::GenericRegister{T,N}) where T where N; return T; end
function register_size(reg::GenericRegister{T,N}) where T where N; return N; end

end #QJuliaRegisters
