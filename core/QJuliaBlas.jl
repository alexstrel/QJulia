module QJuliaBlas

import QJuliaIntrinsics

using MPI

@inline function qjulia_blas_info()
  println("Experimental: executing blas routines in ", Threads.nthreads(), " threads.")
end

#create function/type alias
#SSE
m128d   = QJuliaIntrinsics.m128d
m128    = QJuliaIntrinsics.m128
#AVX/AVX2
m256d   = QJuliaIntrinsics.m256d
m256    = QJuliaIntrinsics.m256
#AVX3
m512d   = QJuliaIntrinsics.m512d
m512    = QJuliaIntrinsics.m512

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
Threads.@threads  for i in 1:length(x); y[2i-1] = real(x[i]); y[2i] = imag(x[i]); end
end #convert_c2r

@inline function convert_c2r(y::AbstractArray, x::Vector{Complex{T}}) where T <: AbstractFloat
Threads.@threads  for i in 1:length(x); y[2i-1] = real(x[i]); y[2i] = imag(x[i]); end
end #convert_c2r

@inline function convert_r2c(y::Vector{Complex{T}}, x::Vector{T}) where T <: AbstractFloat
Threads.@threads  for i in 1:length(y); y[i] = x[2i-1] + x[2i]*im; end
end #convert_r2c

@inline function convert_r2c(y::Vector{Complex{T}}, x::AbstractArray) where T <: AbstractFloat
Threads.@threads  for i in 1:length(y); y[i] = x[2i-1] + x[2i]*im; end
end #convert_r2c


@inline function gcpy(y::AbstractArray, x::AbstractArray) 
  if pointer_from_objref(y) == pointer_from_objref(x); return; end
Threads.@threads for i in 1:length(x);  y[i] = x[i]; end
end #gcpy


@inline function gaxpy(a, x::AbstractArray, y::AbstractArray) 
Threads.@threads for i in 1:length(x);  y[i] = mm_mad(a, x[i], y[i]); end
end #gaxpy

@inline function gxpay(x::AbstractArray, a, y::AbstractArray) 
Threads.@threads for i in 1:length(x);  y[i] = mm_mad(a, y[i], x[i]); end
end #gxpay

@inline function gxmy(x::AbstractArray, y::AbstractArray) 
Threads.@threads for i in 1:length(x);  y[i] = mm_sub(x[i], y[i]); end
end #gxmy

@inline function gxpy(x::AbstractArray, y::AbstractArray) 
Threads.@threads for i in 1:length(x);  y[i] = mm_add(x[i], y[i]); end
end #gxpy

@inline function crxpy(x::Vector{Complex{T}}, y::Vector{T}) where T <: AbstractFloat 
Threads.@threads for i in 1:length(x);  y[2i-1] = real(x[i]) + y[2i-1]; y[2i] = imag(x[i]) + y[2i]; end
end #crxpy

#even more generic
@inline function crxpy(x::Vector{Complex{T}}, y::AbstractArray) where T <: AbstractFloat 
Threads.@threads for i in 1:length(x);  y[2i-1] = real(x[i]) + y[2i-1]; y[2i] = imag(x[i]) + y[2i]; end
end #crxpy

@inline function rcxpy(x::Vector{T}, y::Vector{Complex{T}}) where T <: AbstractFloat 
Threads.@threads for i in 1:length(y)  
                    re_y = x[2i-1] + real(y[i])
                    im_y = x[2i  ] + imag(y[i])
                    y[i] = re_y + im_y*im
                  end
end #rcxpy

#First performs the operation y[i] += a*x[i]
#Second performs the operator x[i] -= a*z[i]

@inline function caxpyXmaz(a, x::Vector{T}, y::Vector{T}, z::Vector{T})  where T <: AbstractFloat 

                 complex_len = Int(length(x) / 2)
Threads.@threads for i in 1:complex_len
                    rex = x[2i-1]; imx = x[2i]; rez = z[2i-1]; imz = z[2i]
                    y[2i-1] += (real(a)*rex-imag(a)*imx)
                    y[2i  ] += (real(a)*imx+imag(a)*rex)
                    x[2i-1] -= (real(a)*rez-imag(a)*imz)
                    x[2i  ] -= (real(a)*imz+imag(a)*rez)
                 end
end #caxpyXmaz

#First performs the operation x[i] = x[i] + a*p[i]
#Second performs the operator p[i] = u[i] + b*p[i]

@inline function axpyZpbx(a, p::Vector{T}, x::Vector{T}, u::Vector{T}, b)  where T <: AbstractFloat 

Threads.@threads for i in 1:length(x)
                    x[i] = x[i]+a*p[i] 
                    p[i] = u[i]+b*p[i] 
                 end
end #axpyZpbx



end #QJuliaBlas


