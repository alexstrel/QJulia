module QJuliaIntrinsics

import QJuliaRegisters

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


m256dfma(a,b,c) = ccall("llvm.fma.v4f64", llvmcall, m256d, (m256d, m256d, m256d), a, b, c)
m256fma(a,b,c ) = ccall("llvm.fma.v8f32", llvmcall, m256, (m256, m256, m256), a, b, c)

m512dfma(a,b,c) = ccall("llvm.fma.v8f64", llvmcall, m512d, (m512d, m512d, m512d), a, b, c)
m512fma(a,b,c ) = ccall("llvm.fma.v16f32", llvmcall, m512, (m512, m512, m512), a, b, c)

prefetch(a, i1, i2, i3) = ccall("llvm.prefetch", llvmcall, Cvoid, (Ptr{Cchar}, Int32, Int32, Int32), a, i1, i2, i3)

prefetchT0(a) = mmprefetch(a, Int32(0), Int32(3), Int(1))
prefetchT1(a) = mmprefetch(a, Int32(0), Int32(2), Int(1))
prefetchT2(a) = mmprefetch(a, Int32(0), Int32(1), Int(1))


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
   zmm = m256((a, a, a, a, a, a, a, a))
   return m256fma(zmm,xmm,ymm)
end

@inline function mm_mad(a::T, xmm::m256d, ymm::m256d) where T <: AbstractFloat
   zmm = m256d((a, a, a, a))
   return m256dfma(zmm,xmm,ymm)
end

end #QJuliaIntrinsics
