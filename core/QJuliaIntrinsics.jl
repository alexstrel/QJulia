module QJuliaIntrinsics

#SSE vector regs
const m128  = NTuple{4, VecElement{Float32}}
const m128d = NTuple{2, VecElement{Float64}}
#AVX/AVX2 vector regs
const m256  = NTuple{8, VecElement{Float32}}
const m256d = NTuple{4, VecElement{Float64}}
#AVX3 vector regs
const m512  = NTuple{16, VecElement{Float32}}
const m512d = NTuple{8 , VecElement{Float64}}

@inline function mm_mul(x::T, y::T) where T <:AbstractFloat
   return x * y
end

@inline function mm_mul(xmm::m128, ymm::m128)
  (VecElement(xmm[1].value * ymm[1].value),
   VecElement(xmm[2].value * ymm[2].value),
   VecElement(xmm[3].value * ymm[3].value),
   VecElement(xmm[4].value * ymm[4].value))
end

@inline function mm_mul(xmm::m128d, ymm::m128d)
  (VecElement(xmm[1].value * ymm[1].value),
   VecElement(xmm[2].value * ymm[2].value))
end

@inline function mm_mul(xmm::m256, ymm::m256)
  (VecElement(xmm[1].value * ymm[1].value),
   VecElement(xmm[2].value * ymm[2].value),
   VecElement(xmm[3].value * ymm[3].value),
   VecElement(xmm[4].value * ymm[4].value),
   VecElement(xmm[5].value * ymm[5].value),
   VecElement(xmm[6].value * ymm[6].value),
   VecElement(xmm[7].value * ymm[7].value),
   VecElement(xmm[8].value * ymm[8].value))
end

@inline function mm_mul(xmm::m256d, ymm::m256d)
  (VecElement(xmm[1].value * ymm[1].value),
   VecElement(xmm[2].value * ymm[2].value),
   VecElement(xmm[3].value * ymm[3].value),
   VecElement(xmm[4].value * ymm[4].value))
end


@inline function mm_add(x::T, y::T) where T <:AbstractFloat
   return x + y
end

@inline function mm_add(xmm::m128, ymm::m128)
  (VecElement(xmm[1].value + ymm[1].value),
   VecElement(xmm[2].value + ymm[2].value),
   VecElement(xmm[3].value + ymm[3].value),
   VecElement(xmm[4].value + ymm[4].value))
end

@inline function mm_add(xmm::m128d, ymm::m128d)
  (VecElement(xmm[1].value + ymm[1].value),
   VecElement(xmm[2].value + ymm[2].value))
end

@inline function mm_add(xmm::m256, ymm::m256)
  (VecElement(xmm[1].value + ymm[1].value),
   VecElement(xmm[2].value + ymm[2].value),
   VecElement(xmm[3].value + ymm[3].value),
   VecElement(xmm[4].value + ymm[4].value),
   VecElement(xmm[5].value + ymm[5].value),
   VecElement(xmm[6].value + ymm[6].value),
   VecElement(xmm[7].value + ymm[7].value),
   VecElement(xmm[8].value + ymm[8].value))
end

@inline function mm_add(xmm::m256d, ymm::m256d)
  (VecElement(xmm[1].value + ymm[1].value),
   VecElement(xmm[2].value + ymm[2].value),
   VecElement(xmm[3].value + ymm[3].value),
   VecElement(xmm[4].value + ymm[4].value))
end

@inline function mm_sub(x::T, y::T) where T <:AbstractFloat
   return x - y
end

@inline function mm_sub(xmm::m128, ymm::m128)
  (VecElement(xmm[1].value - ymm[1].value),
   VecElement(xmm[2].value - ymm[2].value),
   VecElement(xmm[3].value - ymm[3].value),
   VecElement(xmm[4].value - ymm[4].value))
end

@inline function mm_sub(xmm::m128d, ymm::m128d)
  (VecElement(xmm[1].value - ymm[1].value),
   VecElement(xmm[2].value - ymm[2].value))
end

@inline function mm_sub(xmm::m256, ymm::m256)
  (VecElement(xmm[1].value - ymm[1].value),
   VecElement(xmm[2].value - ymm[2].value),
   VecElement(xmm[3].value - ymm[3].value),
   VecElement(xmm[4].value - ymm[4].value),
   VecElement(xmm[5].value - ymm[5].value),
   VecElement(xmm[6].value - ymm[6].value),
   VecElement(xmm[7].value - ymm[7].value),
   VecElement(xmm[8].value - ymm[8].value))
end

@inline function mm_sub(xmm::m256d, ymm::m256d)
  (VecElement(xmm[1].value - ymm[1].value),
   VecElement(xmm[2].value - ymm[2].value),
   VecElement(xmm[3].value - ymm[3].value),
   VecElement(xmm[4].value - ymm[4].value))
end

@inline function mm_mad(a::T, x::T, y::T) where T <:AbstractFloat
   return a * x + y
end

@inline function mm_mad(a::T, xmm::m128, ymm::m128) where T <: AbstractFloat
  (VecElement(a * xmm[1].value + ymm[1].value),
   VecElement(a * xmm[2].value + ymm[2].value),
   VecElement(a * xmm[3].value + ymm[3].value),
   VecElement(a * xmm[4].value + ymm[4].value))
end

@inline function mm_mad(a::T, xmm::m128d, ymm::m128d) where T <: AbstractFloat
  (VecElement(a * xmm[1].value + ymm[1].value),
   VecElement(a * xmm[2].value + ymm[2].value))
end

@inline function mm_mad(a::T, xmm::m256, ymm::m256) where T <: AbstractFloat
  (VecElement(a * xmm[1].value + ymm[1].value),
   VecElement(a * xmm[2].value + ymm[2].value),
   VecElement(a * xmm[3].value + ymm[3].value),
   VecElement(a * xmm[4].value + ymm[4].value),
   VecElement(a * xmm[5].value + ymm[5].value),
   VecElement(a * xmm[6].value + ymm[6].value),
   VecElement(a * xmm[7].value + ymm[7].value),
   VecElement(a * xmm[8].value + ymm[8].value))
end

@inline function mm_mad(a::T, xmm::m256d, ymm::m256d) where T <: AbstractFloat
  (VecElement(a * xmm[1].value + ymm[1].value),
   VecElement(a * xmm[2].value + ymm[2].value),
   VecElement(a * xmm[3].value + ymm[3].value),
   VecElement(a * xmm[4].value + ymm[4].value))
end

end #QJuliaIntrinsics


