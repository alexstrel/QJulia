module QJuliaReduce

import QJuliaIntrinsics

using MPI

blas_global_reduction = true

@inline function set_blas_global_reduction(flag::Bool)
  global blas_global_reduction = flag
end

@inline function reset_blas_global_reduction()
  global blas_global_reduction = true
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

@inline function mm_dot(x::Vector{m256d}, y::Vector{m256d}) 
                 global res = m256d(ntuple(i->0.0, 4)) 
                 for i in 1:length(x)  
                   a   = mm_mul(x[i], y[i])
                   res = mm_add(res,a)
                 end
                 
                 send_buff = [res[1].value, res[2].value, res[3].value, res[4].value]
                 if (blas_global_reduction == true)
                   recv_buff = zeros(Cdouble, 4)
                   MPI.Allreduce!(send_buff, recv_buff, MPI.SUM, MPI.COMM_WORLD)
                 else
                   recv_buff = send_buff
                 end

                 return (recv_buff[1]+recv_buff[2]+recv_buff[3]+recv_buff[4]) 
end #mm_dot

@inline function gnorm2(x::Vector{T})  where T <: AbstractFloat 
                 global res = 0.0 
                 for i in 1:length(x); res += x[i] * x[i]; end

                 if (blas_global_reduction == true)
                   val = MPI.Allreduce(res, MPI.SUM, MPI.COMM_WORLD)
                 else
                   val = res
                 end

                 return val 
end #gnorm2

@inline function gnorm2(x::Vector{Complex{T}})  where T <: AbstractFloat 
                 global res = 0.0 
                 for i in 1:length(x); res += (real(x[i]) * real(x[i]) + imag(x[i]) * imag(x[i])); end

                 if (blas_global_reduction == true)
                   val = MPI.Allreduce(res, MPI.SUM, MPI.COMM_WORLD)
                 else
                   val = res
                 end

                 return val 
end #gnorm2

@inline function gnorm2(x::AbstractArray)  
                 global res = 0.0 
                 for i in 1:length(x); res += abs2(x[i]); end

                 if (blas_global_reduction == true)
                   val = MPI.Allreduce(res, MPI.SUM, MPI.COMM_WORLD)
                 else
                   val = res
                 end

                 return res 
end #gnorm2

@inline function cDotProductNormX(x::Vector{T}, y::Vector{T})  where T <: AbstractFloat 
                 global cres = 0.0+0.0im 
                 global rres = 0.0 
                 complex_len = Int(length(x) / 2)
                 for i in 1:complex_len
                    conjcx    = x[2i-1]-x[2i]*im 
                    cy        = y[2i-1]+y[2i]*im 
                    cres += (conjcx * cy)
                    rres += abs2(conjcx)
                 end

                 send_buff = [real(cres), imag(cres), rres] 

                 if blas_global_reduction == true
                   recv_buff = zeros(Cdouble, 3)
                   MPI.Allreduce!(send_buff, recv_buff, MPI.SUM, MPI.COMM_WORLD)
                 else
                   recv_buff = send_buff
                 end

                 return (Complex{T}(recv_buff[1], recv_buff[2]), recv_buff[3])
end #cDotProductNormX

@inline function reDotProduct(x::Vector{T}, y::Vector{T})  where T <: AbstractFloat 
                 global res = 0.0 
                 for i in 1:length(x); res += x[i]*y[i]; end

                 if (blas_global_reduction == true)
                   val = MPI.Allreduce(res, MPI.SUM, MPI.COMM_WORLD)
                 else
                   val = res
                 end

                 return val 
end #reDotProduct


end #QJuliaReduce


