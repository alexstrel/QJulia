#!/usr/bin/env julia

module QJuliaSUNtest

#load path to qjulia home directory
push!(LOAD_PATH, joinpath(@__DIR__, "../core"))
push!(LOAD_PATH, joinpath(@__DIR__, "../main/fields"))
push!(LOAD_PATH, @__DIR__)

import QJuliaRegisters
import QJuliaEnums
import QJuliaGrid
import QJuliaFields
import QJuliaCUDAFields

using CUDAdrv
using CUDAnative
using CuArrays
using BenchmarkTools
using StaticArrays

#Float2 version
@inline function axpy_(a, x::Complex{T}, y::Complex{T})::Complex{T} where T <: AbstractFloat

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

 @inline function axpy_(a::T, x::NTuple{2, VecElement{T}}, y::NTuple{2, VecElement{T}})::NTuple{2, VecElement{T}} where T <: AbstractFloat

   res1 = CUDAnative.fma(a, x[1].value, y[1].value)
   res2 = CUDAnative.fma(a, x[2].value, y[2].value)

   return NTuple{2, VecElement{T}}((res1, res2))
 end

 @inline function caxpy_(a::Complex{T}, x::NTuple{2, VecElement{T}}, y::NTuple{2, VecElement{T}})::NTuple{2, VecElement{T}} where T <: AbstractFloat

   rres = CUDAnative.fma(+real(a), x[1].value, y[1].value)
   rres = CUDAnative.fma(-imag(a), x[2].value, rres      )

   ires = CUDAnative.fma(+real(a), x[2].value, y[2].value)
   ires = CUDAnative.fma(+imag(a), x[1].value, ires      )

   return NTuple{2, VecElement{T}}((res1, res2))
 end


 @inline function axpy_(a::NTuple{2, VecElement{T}}, x::NTuple{2, VecElement{T}}, y::NTuple{2, VecElement{T}})::NTuple{2, VecElement{T}} where T <: AbstractFloat
   return (VecElement(CUDAnative.fma(a[1].value, x[1].value, y[1].value)), VecElement(CUDAnative.fma(a[2].value, x[2].value, y[2].value)))
   #return (VecElement(a[1].value*x[1].value + y[1].value), VecElement(a[2].value*x[1].value + y[2].value))
 end

 @inline function caxpy_(a::NTuple{2, VecElement{T}}, x::NTuple{2, VecElement{T}}, y::NTuple{2, VecElement{T}})::NTuple{2, VecElement{T}} where T <: AbstractFloat
   res  = (VecElement(CUDAnative.fma(+a[1].value, x[1].value, y[1].value)),
           VecElement(CUDAnative.fma(+a[1].value, x[2].value, y[2].value)))
   return (VecElement(CUDAnative.fma(-a[2].value, x[2].value, res[1].value)),
           VecElement(CUDAnative.fma(+a[2].value, x[1].value, res[2].value)))
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



function axpy(a, xaccessor, yaccessor)
    xcb = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # get data arrays from the accessors
    xdata = xaccessor[1]; ydata = yaccessor[1]
    args  = xaccessor[2]
    #
    if xcb > args.length; return; end

#    if (Int64(blockIdx().x) == 1 && threadIdx().x == 1)
#      for s in 1:args.nSpin
#        for c in 1:args.nColor
#          @cuprintf("color %d  ", c)
#          @cuprintf("spin  %d\n", s)
#        end
#      end
#      @cuprintf("Info block %ld, thread %ld : %d  (length), %d \n", Int64(blockIdx().x), Int64(threadIdx().x), args.length, args.length)
#      @cuprintf("Info block %ld, thread %ld : %d  (colors) \n", Int64(blockIdx().x), Int64(threadIdx().x), args.nColor)
#    end

    y_site_view = view(ydata,xcb,:,:,1)
    x_site_view = view(xdata,xcb,:,:,1)

    for s in 1:args.nSpin
      for c in 1:args.nColor
        y_site_view[c, s] = axpy_(a, x_site_view[c, s], y_site_view[c, s])
      end
    end

  #  if (Int64(blockIdx().x) == 1 && threadIdx().x == 4)
  #    for s in 1:args.nSpin
  #      for c in 1:args.nColor
  #        @cuprintf("Info block %ld, thread %ld : %le, %le!\n", Int64(blockIdx().x), Int64(threadIdx().x), x_site_view[c,s][1].value, x_site_view[c,s][2].value)
  #      end
  #    end
  #  end
    return
end

function set_value(x, a)
    xcb = (blockIdx().x-1) * blockDim().x + threadIdx().x

    (xlength, xcolors, xspins, xvecs) = size(x)

    if xcb > xlength; return; end


    xview = view(x,:,:,:,1)

    for s in 1:xspins
      for c in 1:xcolors
        xview[xcb, c, s] =  NTuple{2, Float32}((Float32(s), Float32(c)))
      end
    end

    return
end


function sun_test(x, y)
    xcb = (blockIdx().x-1) * blockDim().x + threadIdx().x
    #    wid, lane = fldmod1(threadIdx().x, CUDAnative.warpsize())
    lid = Int64(threadIdx().x)

    (nElems, nCols, nRows, nDirs) = size(x)

    if xcb > nElems; return; end

    dir = 1

    y_site_view = view(y,xcb,:,:,dir)
    x_site_view = view(x,xcb,:,:,dir)

    #shared_dims = blockDim().x*nCols*nRows*get_regN(x_site_view[1,1])
    #tmp = @cuDynamicSharedMem(get_regT(x_site_view[1,1]), shared_dims)
    msize = nCols*nRows*4 #get_regN(x_site_view[1,1])

    shared_dims = blockDim().x*msize
    tmp = @cuDynamicSharedMem(Float32, shared_dims)

    #if (Int64(blockIdx().x) == 1)
    #  @cuprintf("Info block %ld, thread %ld : %d  (wid), %d (lane), %d (warp size) \n", Int64(blockIdx().x), Int64(threadIdx().x), wid, lane, CUDAnative.warpsize())
    #end


    for r in 1:nRows
      for c in 1:nCols
        temp = NTuple{4, VecElement{Float32}}((0.0f0, 0.0f0, 0.0f0, 0.0f0))
        for j in 1:nCols
          temp = caxpy_(x_site_view[r, j], y_site_view[j, c], temp)
        end
        @inbounds tmp[msize*(lid-1)+(r+c*3)*2+0] = temp[1].value
        @inbounds tmp[msize*(lid-1)+(r+c*3)*2+1] = temp[2].value
        @inbounds tmp[msize*(lid-1)+(r+c*3)*2+2] = temp[3].value
        @inbounds tmp[msize*(lid-1)+(r+c*3)*2+3] = temp[4].value
      end
    end

    for r in 1:nRows
      for c in 1:nCols
        temp = NTuple{4, VecElement{Float32}}((tmp[msize*(lid-1)+(r+c*3)*2+0], tmp[msize*(lid-1)+(r+c*3)*2+1], tmp[msize*(lid-1)+(r+c*3)*2+2], tmp[msize*(lid-1)+(r+c*3)*2+3]))
        y_site_view[r, c] = temp
      end
    end

#    if (Int64(blockIdx().x) == 1 && threadIdx().x == 4)
#      for c in 1:xrows
#        for r in 1:xcols
#          @cuprintf("Info block %ld, thread %ld : %le, %le!\n", Int64(blockIdx().x), Int64(threadIdx().x), y_site_view[r,c][1].value, y_site_view[r,c][2].value)
#        end
#      end
#    end

    return
end

function sun_test_pure_arg_templ(x::CuDeviceArray{NTuple{N,VecElement{T}},4}, y::CuDeviceArray{NTuple{N,VecElement{T}},4}) where T <: AbstractFloat where N
    xcb = (blockIdx().x-1) * blockDim().x + threadIdx().x
    #    wid, lane = fldmod1(threadIdx().x, CUDAnative.warpsize())
    tid  = Int64(threadIdx().x)
    bdim = Int64(blockDim().x)

    (nElems, nCols, nRows, nDirs) = size(x)

    if xcb > nElems; return; end

    dir = 1

    y_site_view = view(y,xcb,:,:,dir)
    x_site_view = view(x,xcb,:,:,dir)

    #shared_dims = blockDim().x*nCols*nRows*get_regN(x_site_view[1,1])
    #tmp = @cuDynamicSharedMem(get_regT(x_site_view[1,1]), shared_dims)

    shared_dims = bdim*nCols*nRows*N
    tmp = @cuDynamicSharedMem(T, shared_dims)

    #if (Int64(blockIdx().x) == 1)
    #  @cuprintf("Info block %ld, thread %ld : %d  (wid), %d (lane), %d (warp size) \n", Int64(blockIdx().x), Int64(threadIdx().x), wid, lane, CUDAnative.warpsize())
    #end


    for r in 1:nRows
      for c in 1:nCols
        temp = NTuple{N, VecElement{T}}(ntuple(i->T(0.0), N))
        for j in 1:nCols
          temp = caxpy_(x_site_view[r, j], y_site_view[j, c], temp)
        end
        offset = tid + ((r-1)+(c-1)*3)*bdim
        for l in 1:N
          @inbounds tmp[offset+(l-1)*9*bdim] = temp[l].value
        end
      end
    end

    for r in 1:nRows
      for c in 1:nCols
        offset = tid + ((r-1)+(c-1)*3)*bdim
        temp = N == 2 ? NTuple{N, VecElement{T}}( (tmp[offset+0*9*bdim], tmp[offset+1*9*bdim])) : NTuple{N, VecElement{T}}( (tmp[offset+0*9*bdim], tmp[offset+1*9*bdim],tmp[offset+2*9*bdim], tmp[offset+3*9*bdim]) )
        @inbounds y_site_view[r, c] = temp
      end
    end

#    if (Int64(blockIdx().x) == 1 && threadIdx().x == 4)
#      for c in 1:xrows
#        for r in 1:xcols
#          @cuprintf("Info block %ld, thread %ld : %le, %le!\n", Int64(blockIdx().x), Int64(threadIdx().x), y_site_view[r,c][1].value, y_site_view[r,c][2].value)
#        end
#      end
#    end

    return
end


function sun_extended(z::CuDeviceArray{NTuple{N, VecElement{T}},4},  x::CuDeviceArray{NTuple{N,VecElement{T}},4}, y::CuDeviceArray{NTuple{N,VecElement{T}},4}) where T <: AbstractFloat where N
    xcb = (blockIdx().x-1) * blockDim().x + threadIdx().x

    (nElems, nCols, nRows, nDirs) = size(x)

    if xcb > nElems; return; end

    dir = 1

    z_site_view = view(z,xcb,:,:,dir)
    y_site_view = view(y,xcb,:,:,dir)
    x_site_view = view(x,xcb,:,:,dir)

    #if (Int64(blockIdx().x) == 1)
    #  @cuprintf("Info block %ld, thread %ld : %d  (wid), %d (lane), %d (warp size) \n", Int64(blockIdx().x), Int64(threadIdx().x), wid, lane, CUDAnative.warpsize())
    #end

    for r in 1:nRows
      for c in 1:nCols
        temp = NTuple{N, VecElement{T}}(ntuple(i->T(0.0), N))
        for j in 1:nCols
          temp = caxpy_(x_site_view[r, j], y_site_view[j, c], temp)
        end
        z_site_view[r,c] = temp
      end
    end

    return
 end



function sun_test_templ(x::Tuple{CuDeviceArray{NTuple{N,VecElement{T}},4},QJuliaFields.AccessorArgs}, y::Tuple{CuDeviceArray{NTuple{N,VecElement{T}},4},QJuliaFields.AccessorArgs}) where T <: AbstractFloat where N
    xcb = (blockIdx().x-1) * blockDim().x + threadIdx().x
    #    wid, lane = fldmod1(threadIdx().x, CUDAnative.warpsize())
    tid  = Int64(threadIdx().x)
    bdim = Int64(blockDim().x)

    (nElems, nCols, nRows, nDirs) = size(x[1])

    if xcb > nElems; return; end

    dir = 1

    y_site_view = view(y[1],xcb,:,:,dir)
    x_site_view = view(x[1],xcb,:,:,dir)

    #shared_dims = blockDim().x*nCols*nRows*get_regN(x_site_view[1,1])
    #tmp = @cuDynamicSharedMem(get_regT(x_site_view[1,1]), shared_dims)

    shared_dims = bdim*nCols*nRows*N
    tmp = @cuDynamicSharedMem(T, shared_dims)

    #if (Int64(blockIdx().x) == 1)
    #  @cuprintf("Info block %ld, thread %ld : %d  (wid), %d (lane), %d (warp size) \n", Int64(blockIdx().x), Int64(threadIdx().x), wid, lane, CUDAnative.warpsize())
    #end


    for r in 1:nRows
      for c in 1:nCols
        temp = NTuple{N, VecElement{T}}(ntuple(i->T(0.0), N))
        for j in 1:nCols
          temp = caxpy_(x_site_view[r, j], y_site_view[j, c], temp)
        end
        offset = tid + ((r-1)+(c-1)*3)*bdim
        for l in 1:N
          @inbounds tmp[offset+(l-1)*9*bdim] = temp[l].value
        end
      end
    end

    for r in 1:nRows
      for c in 1:nCols
        offset = tid + ((r-1)+(c-1)*3)*bdim
        temp = N == 2 ? NTuple{N, VecElement{T}}( (tmp[offset+0*9*bdim], tmp[offset+1*9*bdim])) : NTuple{N, VecElement{T}}( (tmp[offset+0*9*bdim], tmp[offset+1*9*bdim],tmp[offset+2*9*bdim], tmp[offset+3*9*bdim]) )
        y_site_view[r, c] = temp
      end
    end

#    if (Int64(blockIdx().x) == 1 && threadIdx().x == 4)
#      for c in 1:xrows
#        for r in 1:xcols
#          @cuprintf("Info block %ld, thread %ld : %le, %le!\n", Int64(blockIdx().x), Int64(threadIdx().x), y_site_view[r,c][1].value, y_site_view[r,c][2].value)
#        end
#      end
#    end

    return
end



  const  nthreads = 128

  #data_type = ComplexF32
  #data_type = QJuliaRegisters.float2
  data_type = QJuliaRegisters.float4

  const N = 32
  println("Test lattice size ::" , N)

  csGrid   = QJuliaGrid.QJuliaGridDescr_qj{data_type}(QJuliaEnums.QJULIA_CUDA_FIELD_LOCATION, 0, (N,N,N,N,))

  # compilation time
  gaugeParam  = QJuliaGrid.CreateGaugeParams( csGrid )

  cuda_sun_m1 = QJuliaCUDAFields.CreateGenericField( gaugeParam )
  cuda_sun_m2 = QJuliaCUDAFields.CreateGenericField( gaugeParam )
  cuda_sun_m3 = QJuliaCUDAFields.CreateGenericField( gaugeParam )

  println("\n=====GAUGE FIELD INFO=====")
  QJuliaFields.field_info(cuda_sun_m1)

  println("====================")

  cuda_sun_m1_accessor = QJuliaFields.create_field_accessor(cuda_sun_m1)
  cuda_sun_m2_accessor = QJuliaFields.create_field_accessor(cuda_sun_m2)
  cuda_sun_m3_accessor = QJuliaFields.create_field_accessor(cuda_sun_m3)

  accessor_dims = size(cuda_sun_m1_accessor[1])

  println("  ::: ", typeof(cuda_sun_m1_accessor))

  len = accessor_dims[1]

  nblocks= ceil.(Int, len ./ nthreads)
  # calculate size of dynamic shared memory
  regN    = QJuliaRegisters.register_size(typeof(cuda_sun_m1_accessor[1][1]))
  regT    = QJuliaRegisters.register_type(typeof(cuda_sun_m1_accessor[1][1]))

  ncolors = accessor_dims[2]

  shmem_bytes = ncolors*ncolors*regN* nthreads * sizeof(regT)

  println("Execution config for length ", len, " : ", nthreads , " threads, ", nblocks, " blocks , shared mem ", shmem_bytes )

  # compilation options
#code inspections:
#@device_code_typed
#@device_code_llvm
#@device_code_ptx
#@device_code_sass
  @device_code_ptx @cuda blocks=nblocks threads=nthreads shmem=shmem_bytes sun_test_pure_arg_templ(cuda_sun_m1_accessor[1], cuda_sun_m2_accessor[1])
  #@time @cuda blocks=nblocks threads=nthreads shmem=shmem_bytes sun_test_templ(cuda_sun_m1_accessor, cuda_sun_m2_accessor)
  #@device_code_ptx @cuda blocks=nblocks threads=nthreads shmem=shmem_bytes sun_extended(cuda_sun_m3_accessor[1],  cuda_sun_m1_accessor[1], cuda_sun_m2_accessor[1])

  # execution time
  niters = 10000
  secs = CUDAdrv.@elapsed begin # begin profile
  for i in 1:niters
    @cuda blocks=nblocks threads=nthreads shmem=shmem_bytes sun_test_pure_arg_templ(cuda_sun_m1_accessor[1], cuda_sun_m2_accessor[1])
    #@cuda blocks=nblocks threads=nthreads shmem=shmem_bytes sun_test_templ(cuda_sun_m1_accessor, cuda_sun_m2_accessor)
    #@cuda blocks=nblocks threads=nthreads shmem=shmem_bytes sun_extended(cuda_sun_m3_accessor[1], cuda_sun_m1_accessor[1], cuda_sun_m2_accessor[1])
  end
  end # end profile

  vol  = regN*(len / 2.0) #complex array length
  ncol = 3
  bytes_per_link_dir = 3.0*vol*length(cuda_sun_m1.v) / 4.0
  footprint_per_link_dir = 2.0*vol*length(cuda_sun_m1.v) / 4.0
  flops = (8.0*ncol - 2.0)*ncol*ncol*vol

  secs_per_iter = (secs / Float64(niters))

  println("Bytes :: ", bytes_per_link_dir, " footprint :: ", footprint_per_link_dir, " flops ",  flops / secs_per_iter)


end # module
