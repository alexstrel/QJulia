module QJuliaBlas

import QJuliaRegisters
import QJuliaIntrinsics

using Base.Threads

@inline function qjulia_blas_info()
  println("Experimental: executing blas routines in ", Threads.nthreads(), " threads.")
end

#create function/type alias
#SSE
m128d   = QJuliaRegisters.double2
m128    = QJuliaRegisters.float4
#AVX/AVX2
m256d   = QJuliaRegisters.double4
m256    = QJuliaRegisters.float8
#AVX3
m512d   = QJuliaRegisters.double8
m512    = QJuliaRegisters.float16

mm_mul  = QJuliaIntrinsics.mm_mul
mm_add  = QJuliaIntrinsics.mm_add
mm_sub  = QJuliaIntrinsics.mm_sub
mm_mad  = QJuliaIntrinsics.mm_mad

#collection of generic blas operations (e.g., on vector registers etc.)

@inline function convert_c2r(y::Vector{m256d}, x::Vector{Complex{T}}) where T <: AbstractFloat
  for i in 1:length(y) 
    c=[real(x[2i-1]), imag(x[2i-1]), real(x[2i]), imag(x[2i]) ] 
    y[i] = m256d(ntuple(i->c[i],4))
  end
end #convert_c2r

@inline function convert_c2r(y::Vector{m256}, x::Vector{Complex{T}}) where T <: AbstractFloat
  for i in 1:length(y) 
    c=[real(x[4i-3]), imag(x[4i-3]), real(x[4i-2]), imag(x[4i-2]), real(x[4i-1]), imag(x[4i-1]), real(x[4i]), imag(x[4i]) ] 
    y[i] = m256(ntuple(i->c[i],8))
  end
end #convert_c2r

@inline function convert_c2r(y::Vector{T}, x::Vector{Complex{T}}) where T <: AbstractFloat
@threads  for i in 1:length(x); y[2i-1] = real(x[i]); y[2i] = imag(x[i]); end
end #convert_c2r

@inline function convert_c2r(y::AbstractArray, x::Vector{Complex{T}}) where T <: AbstractFloat
@threads  for i in 1:length(x); y[2i-1] = real(x[i]); y[2i] = imag(x[i]); end
end #convert_c2r

@inline function convert_r2c(y::Vector{Complex{T}}, x::Vector{T}) where T <: AbstractFloat
@threads  for i in 1:length(y); y[i] = x[2i-1] + x[2i]*im; end
end #convert_r2c

@inline function convert_r2c(y::Vector{Complex{T}}, x::AbstractArray) where T <: AbstractFloat
@threads  for i in 1:length(y); y[i] = x[2i-1] + x[2i]*im; end
end #convert_r2c


@inline function cpy(y::AbstractArray, x::AbstractArray) 
  if pointer_from_objref(y) == pointer_from_objref(x); return; end
@threads for i in 1:length(x);  y[i] = x[i]; end
end #cpy


@inline function axpy(a, x::AbstractArray, y::AbstractArray) 
@threads for i in 1:length(x);  y[i] = mm_mad(a, x[i], y[i]); end
end #axpy

@inline function xpay(x::AbstractArray, a, y::AbstractArray) 
@threads for i in 1:length(x);  y[i] = mm_mad(a, y[i], x[i]); end
end #xpay

@inline function xmy(x::AbstractArray, y::AbstractArray) 
@threads for i in 1:length(x);  y[i] = mm_sub(x[i], y[i]); end
end #xmy

@inline function xpy(x::AbstractArray, y::AbstractArray) 
@threads for i in 1:length(x);  y[i] = mm_add(x[i], y[i]); end
end #xpy

#First performs the operation y[i] += a*x[i]
#Second performs the operator x[i] -= a*z[i]

@inline function caxpyXmaz(a, x::Vector{T}, y::Vector{T}, z::Vector{T})  where T <: AbstractFloat 

         cx = view(reinterpret(Complex{T}, x), :)
         cy = view(reinterpret(Complex{T}, y), :)
         cz = view(reinterpret(Complex{T}, z), :)

@threads for i in 1:length(cx)
           cy[i]  += a*cx[i]
           cx[i]  -= a*cz[i]
         end
end #caxpyXmaz

@inline function axpyXmaz(a, x::Vector{T}, y::Vector{T}, z::Vector{T})  where T <: AbstractFloat

@threads for i in 1:length(x)
		y[i]  += a*x[i]
		x[i]  -= a*z[i]
	end   

end #axpyXmaz


#First performs the operation x[i] = x[i] + a*p[i]
#Second performs the operator p[i] = u[i] + b*p[i]

@inline function axpyZpbx(a, p::Vector{T}, x::Vector{T}, u::Vector{T}, b)  where T <: AbstractFloat 

@threads for i in 1:length(x)
           x[i] = x[i]+a*p[i] 
           p[i] = u[i]+b*p[i] 
         end
end #axpyZpbx

end #QJuliaBlas


