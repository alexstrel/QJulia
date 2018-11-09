module QJuliaBlas

#export JULIA_NUM_THREADS=4

@inline function qjulia_blas_info()
  println("Experimental: executing blas routines in ", Threads.nthreads(), " threads.")
end

#SSE vector regs
const m128  = NTuple{4, VecElement{Float32}}
const m128d = NTuple{2, VecElement{Float64}}
#AVX/AVX2 vector regs
const m256  = NTuple{8, VecElement{Float32}}
const m256d = NTuple{4, VecElement{Float64}}
#AVX3 vector regs
#const m512  = NTuple{16, VecElement{Float32}}
#const m512d = NTuple{8 , VecElement{Float64}}


@inline function mm_mul(_x::T, _y::T) where T <:AbstractFloat
   return _x * _y
end

#/usr/lib/x86_64-linux-gnu/libc.so
#@inline function ymmadd(_x::m256d, _y::m256d)
#  ccall((:_mm256_add_pd, "libc"), m256d, (m256d, m256d), _x, _y)
#end

@inline function mm_mul(_x::m128, _y::m128)
  (VecElement(_x[1].value * _y[1].value),
   VecElement(_x[2].value * _y[2].value),
   VecElement(_x[3].value * _y[3].value),
   VecElement(_x[4].value * _y[4].value))
end

@inline function mm_mul(_x::m128d, _y::m128d)
  (VecElement(_x[1].value * _y[1].value),
   VecElement(_x[2].value * _y[2].value))
end

@inline function mm_mul(_x::m256, _y::m256)
  (VecElement(_x[1].value * _y[1].value),
   VecElement(_x[2].value * _y[2].value),
   VecElement(_x[3].value * _y[3].value),
   VecElement(_x[4].value * _y[4].value),
   VecElement(_x[5].value * _y[5].value),
   VecElement(_x[6].value * _y[6].value),
   VecElement(_x[7].value * _y[7].value),
   VecElement(_x[8].value * _y[8].value))
end

@inline function mm_mul(_x::m256d, _y::m256d)
  (VecElement(_x[1].value * _y[1].value),
   VecElement(_x[2].value * _y[2].value),
   VecElement(_x[3].value * _y[3].value),
   VecElement(_x[4].value * _y[4].value))
end


@inline function mm_add(_x::T, _y::T) where T <:AbstractFloat
   return _x + _y
end

#/usr/lib/x86_64-linux-gnu/libc.so
#@inline function ymmadd(_x::m256d, _y::m256d)
#  ccall((:_mm256_add_pd, "libc"), m256d, (m256d, m256d), _x, _y)
#end

@inline function mm_add(_x::m128, _y::m128)
  (VecElement(_x[1].value + _y[1].value),
   VecElement(_x[2].value + _y[2].value),
   VecElement(_x[3].value + _y[3].value),
   VecElement(_x[4].value + _y[4].value))
end

@inline function mm_add(_x::m128d, _y::m128d)
  (VecElement(_x[1].value + _y[1].value),
   VecElement(_x[2].value + _y[2].value))
end

@inline function mm_add(_x::m256, _y::m256)
  (VecElement(_x[1].value + _y[1].value),
   VecElement(_x[2].value + _y[2].value),
   VecElement(_x[3].value + _y[3].value),
   VecElement(_x[4].value + _y[4].value),
   VecElement(_x[5].value + _y[5].value),
   VecElement(_x[6].value + _y[6].value),
   VecElement(_x[7].value + _y[7].value),
   VecElement(_x[8].value + _y[8].value))
end

@inline function mm_add(_x::m256d, _y::m256d)
  (VecElement(_x[1].value + _y[1].value),
   VecElement(_x[2].value + _y[2].value),
   VecElement(_x[3].value + _y[3].value),
   VecElement(_x[4].value + _y[4].value))
end

@inline function mm_sub(_x::T, _y::T) where T <:AbstractFloat
   return _x - _y
end

@inline function mm_sub(_x::m128, _y::m128)
  (VecElement(_x[1].value - _y[1].value),
   VecElement(_x[2].value - _y[2].value),
   VecElement(_x[3].value - _y[3].value),
   VecElement(_x[4].value - _y[4].value))
end

@inline function mm_sub(_x::m128d, _y::m128d)
  (VecElement(_x[1].value - _y[1].value),
   VecElement(_x[2].value - _y[2].value))
end

@inline function mm_sub(_x::m256, _y::m256)
  (VecElement(_x[1].value - _y[1].value),
   VecElement(_x[2].value - _y[2].value),
   VecElement(_x[3].value - _y[3].value),
   VecElement(_x[4].value - _y[4].value),
   VecElement(_x[5].value - _y[5].value),
   VecElement(_x[6].value - _y[6].value),
   VecElement(_x[7].value - _y[7].value),
   VecElement(_x[8].value - _y[8].value))
end

@inline function mm_sub(_x::m256d, _y::m256d)
  (VecElement(_x[1].value - _y[1].value),
   VecElement(_x[2].value - _y[2].value),
   VecElement(_x[3].value - _y[3].value),
   VecElement(_x[4].value - _y[4].value))
end

@inline function mm_mad(a::T, _x::T, _y::T) where T <:AbstractFloat
   return a * _x + _y
end

@inline function mm_mad(a::T, _x::m128, _y::m128) where T <: AbstractFloat
  (VecElement(a * _x[1].value + _y[1].value),
   VecElement(a * _x[2].value + _y[2].value),
   VecElement(a * _x[3].value + _y[3].value),
   VecElement(a * _x[4].value + _y[4].value))
end

@inline function mm_mad(a::T, _x::m128d, _y::m128d) where T <: AbstractFloat
  (VecElement(a * _x[1].value + _y[1].value),
   VecElement(a * _x[2].value + _y[2].value))
end

@inline function mm_mad(a::T, _x::m256, _y::m256) where T <: AbstractFloat
  (VecElement(a * _x[1].value + _y[1].value),
   VecElement(a * _x[2].value + _y[2].value),
   VecElement(a * _x[3].value + _y[3].value),
   VecElement(a * _x[4].value + _y[4].value),
   VecElement(a * _x[5].value + _y[5].value),
   VecElement(a * _x[6].value + _y[6].value),
   VecElement(a * _x[7].value + _y[7].value),
   VecElement(a * _x[8].value + _y[8].value))
end

@inline function mm_mad(a::T, _x::m256d, _y::m256d) where T <: AbstractFloat
  (VecElement(a * _x[1].value + _y[1].value),
   VecElement(a * _x[2].value + _y[2].value),
   VecElement(a * _x[3].value + _y[3].value),
   VecElement(a * _x[4].value + _y[4].value))
end


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

@inline function mm_dot(x::Vector{m256d}, y::Vector{m256d}) 
                 global res = m256d(ntuple(i->0.0, 4)) 
                 for i in 1:length(x)  
                   a   = mm_mul(x[i], y[i])
                   res = mm_add(res,a)
                 end
                 return (res[1].value+res[2].value+res[3].value+res[4].value) 
end #mm_dot

@inline function gnorm2(x::Vector{T})  where T <: AbstractFloat 
                 global res = 0.0 
 for i in 1:length(x); res += x[i] * x[i]; end
                 return res 
end #gnorm2

@inline function gnorm2(x::Vector{Complex{T}})  where T <: AbstractFloat 
                 global res = 0.0 
 for i in 1:length(x); res += (real(x[i]) * real(x[i]) + imag(x[i]) * imag(x[i])); end
                 return res 
end #gnorm2

@inline function gnorm2(x::AbstractArray)  
                 global res = 0.0 
 for i in 1:length(x); res += abs2(x[i]); end
                 return res 
end #gnorm2

end #QJuliaBlas


