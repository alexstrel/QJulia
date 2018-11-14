module QJuliaRegisters

#SSE vector regs
const m128  = NTuple{4, VecElement{Float32}}
const m128d = NTuple{2, VecElement{Float64}}
#AVX/AVX2 vector regs
const m256  = NTuple{8, VecElement{Float32}}
const m256d = NTuple{4, VecElement{Float64}}
#AVX3 vector regs
const m512  = NTuple{16, VecElement{Float32}}
const m512d = NTuple{8 , VecElement{Float64}}

end #QJuliaRegisters


