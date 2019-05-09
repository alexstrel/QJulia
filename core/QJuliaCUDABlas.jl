module QJuliaCUDABlas

push!(LOAD_PATH, @__DIR__)

import QJuliaRegisters

using CUDAnative
using CuArrays

#create function/type alias
double2  = QJuliaRegisters.double2
double4  = QJuliaRegisters.double4
#
float2   = QJuliaRegisters.float2
float4   = QJuliaRegisters.float4

# Real operations
# Native complex values
 @inline function axpy_(a::T, x::Complex{T}, y::Complex{T})::Complex{T} where T <: AbstractFloat

   rres = CUDAnative.fma(a, real(x), real(y))
   ires = CUDAnative.fma(a, imag(x), imag(y))

   return Complex{T}(rres, ires)
 end

 @inline function caxpy_(a::Complex{T}, x::Complex{T}, y::Complex{T})::Complex{T} where T <: AbstractFloat

    rres = CUDAnative.fma(+real(a), real(x), real(y))
    rres = CUDAnative.fma(-imag(a), imag(x), rres   )

    ires = CUDAnative.fma(+imag(a), real(x), imag(y))
    ires = CUDAnative.fma(+real(a), imag(x), ires   )

    return Complex{T}(rres, ires)
  end

# Float2 regs

 @inline function axpy_(a::T, x::NTuple{2, VecElement{T}}, y::NTuple{2, VecElement{T}})::NTuple{2, VecElement{T}} where T <: AbstractFloat

   res1 = CUDAnative.fma(a, x[1].value, y[1].value)
   res2 = CUDAnative.fma(a, x[2].value, y[2].value)

   return NTuple{2, VecElement{T}}((res1, res2))
 end

 @inline function axpy_(a::NTuple{2, VecElement{T}}, x::NTuple{2, VecElement{T}}, y::NTuple{2, VecElement{T}})::NTuple{2, VecElement{T}} where T <: AbstractFloat
   return (VecElement(CUDAnative.fma(a[1].value, x[1].value, y[1].value)), VecElement(CUDAnative.fma(a[2].value, x[2].value, y[2].value)))
 end

 @inline function caxpy_(a::NTuple{2, VecElement{T}}, x::NTuple{2, VecElement{T}}, y::NTuple{2, VecElement{T}})::NTuple{2, VecElement{T}} where T <: AbstractFloat
   res  = (VecElement(CUDAnative.fma(+a[1].value, x[1].value, y[1].value)),
           VecElement(CUDAnative.fma(+a[1].value, x[2].value, y[2].value)))
   return (VecElement(CUDAnative.fma(-a[2].value, x[2].value, res[1].value)),
           VecElement(CUDAnative.fma(+a[2].value, x[1].value, res[2].value)))
 end

# Float4 regs

 @inline function axpy_(a, x::NTuple{4, VecElement{T}}, y::NTuple{4, VecElement{T}})::NTuple{4, VecElement{T}} where T <: AbstractFloat

   res1 = CUDAnative.fma(a, x[1].value, y[1].value)
   res2 = CUDAnative.fma(a, x[2].value, y[2].value)
   res3 = CUDAnative.fma(a, x[3].value, y[3].value)
   res4 = CUDAnative.fma(a, x[4].value, y[4].value)

   return NTuple{4, VecElement{T}}((res1, res2, res3, res4))
 end

 @inline function caxpy_(a::NTuple{4, VecElement{T}}, x::NTuple{4, VecElement{T}}, y::NTuple{4, VecElement{T}})::NTuple{4, VecElement{T}} where T <: AbstractFloat
   res  = (VecElement(CUDAnative.fma(+a[1].value, x[1].value, y[1].value)),
           VecElement(CUDAnative.fma(+a[1].value, x[2].value, y[2].value)),
           VecElement(CUDAnative.fma(+a[3].value, x[3].value, y[3].value)),
           VecElement(CUDAnative.fma(+a[3].value, x[4].value, y[4].value)))
   return (VecElement(CUDAnative.fma(-a[2].value, x[2].value, res[1].value)),
           VecElement(CUDAnative.fma(+a[2].value, x[1].value, res[2].value)),
           VecElement(CUDAnative.fma(-a[4].value, x[4].value, res[3].value)),
           VecElement(CUDAnative.fma(+a[4].value, x[3].value, res[4].value)))
 end

 # a few helper methods:
 @inline function get_regN(meta::NTuple{N, VecElement{T}}) where T <: AbstractFloat where N; return N; end
 @inline function get_regT(meta::NTuple{N, VecElement{T}}) where T <: AbstractFloat where N; return T; end


end #QJuliaCUDABlas
