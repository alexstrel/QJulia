module QJuliaReduce

import QJuliaRegisters
import QJuliaIntrinsics

using MPI

blas_global_reduction = true

@inline function set_blas_global_reduction(flag::Bool)
  global blas_global_reduction = flag
end

@inline function reset_blas_global_reduction()
  global blas_global_reduction = true
end

#SSE
double2   = QJuliaRegisters.double2
float4    = QJuliaRegisters.float4
#AVX/AVX2
double4   = QJuliaRegisters.double4
float8    = QJuliaRegisters.float8
#AVX3
double8   = QJuliaRegisters.double8
float16   = QJuliaRegisters.float16


mm_mul  = QJuliaIntrinsics.mm_mul
mm_add  = QJuliaIntrinsics.mm_add
mm_sub  = QJuliaIntrinsics.mm_sub
mm_mad  = QJuliaIntrinsics.mm_mad

@inline function mm_dot(x::Vector{double4}, y::Vector{double4})
                 local res = double4((0.0, 0.0, 0.0, 0.0))
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
	         #set precision for the big float if necessary.
	         if(T == BigFloat); setprecision(T, 128); end
	         local res::T = 0.0

                 for e in x; res += e*e; end


                 if (blas_global_reduction == true)
		   resd = Float64(res)
                   val  = MPI.Allreduce(resd, MPI.SUM, MPI.COMM_WORLD)
                 else
                   val = res
                 end

                 return val
end #gnorm2

@inline function gnorm2(x::Vector{Complex{T}})  where T <: AbstractFloat
                 local res = 0.0
                 for e in x; res += abs2(e); end

                 if (blas_global_reduction == true)
                   val = MPI.Allreduce(res, MPI.SUM, MPI.COMM_WORLD)
                 else
                   val = res
                 end

                 return val
end #gnorm2

@inline function gnorm2(x::AbstractArray)
                 local res = 0.0
                 for i in 1:length(x); res += abs2(x[i]); end

                 if (blas_global_reduction == true)
                   val = MPI.Allreduce(res, MPI.SUM, MPI.COMM_WORLD)
                 else
                   val = res
                 end

                 return res
end #gnorm2

@inline function cDotProductNormX(x::Vector{T}, y::Vector{T})  where T <: AbstractFloat
                 local cres = 0.0+0.0im
                 local rres = 0.0

                 cx = view(reinterpret(Complex{T}, x), :)
                 cy = view(reinterpret(Complex{T}, y), :)

                 for i in 1:length(cx)
                    conjcx    = conj(cx[i])
                    cres += (conjcx * cy[i])
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

@inline function reDotProductNormX(x::Vector{T}, y::Vector{T})  where T <: AbstractFloat

	local xdty = 0.0
	local xnrm = 0.0

	for i in 1:length(x)
		xdty += (x[i] * y[i])
		xnrm += abs2(x[i])
	end
	send_buff = [xdty, xnrm]
	if blas_global_reduction == true
		recv_buff = zeros(Cdouble, 2)
		MPI.Allreduce!(send_buff, recv_buff, MPI.SUM, MPI.COMM_WORLD)
	else
		recv_buff = send_buff
	end
	return (recv_buff[1], recv_buff[2])
end #reDotProductNormX

@inline function reDotProduct(x::Vector{T}, y::Vector{T})  where T <: AbstractFloat
	             #set precision for the big float if necessary.
	             if(T == BigFloat); setprecision(T, 128); end
                 local res::T = 0.0

                 for i in 1:length(x); res += x[i]*y[i]; end

				 #println("Info : ", sizeof(res), " res = ", res)
				 #resstrn = string(res)
				 #println("String = ", resstrn)
				 #resd1 = parse(Float64, resstrn[1:16]) # do parsing
				 #resd2 = parse(Int64, resstrn[17:32]) # do parsing
				 resd = Float64(res)

                 return (blas_global_reduction == true ?  MPI.Allreduce(resd, MPI.SUM, MPI.COMM_WORLD) : res)
end #reDotProduct

@inline function cDotProduct(x::Vector{T}, y::Vector{T})  where T <: AbstractFloat
                 local res = 0.0+0.0im

                 cx = view(reinterpret(Complex{T}, x), :)
                 cy = view(reinterpret(Complex{T}, y), :)

                 for i in 1:length(cx); res += conj(cx[i]) * cy[i]; end

                 return (blas_global_reduction == true ?  MPI.Allreduce(res, MPI.SUM, MPI.COMM_WORLD) : res)
end #reDotProduct


end #QJuliaReduce
