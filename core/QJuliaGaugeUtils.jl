module QJuliaGaugeUtils

import QJuliaInterface
import QJuliaEnums
import QUDARoutines

debug_constructGaugeField = false

@inline function accumulateConjugateProduct(a::Complex{T}, b::Complex{T}, c::Complex{T}, sign::Float64) where T <: AbstractFloat
   local tmp::Complex{T} = b*c
   a += Complex{T}(sign*real(tmp), -sign*imag(tmp))
   return a
end

# normalize the vector a
function normalize(a::AbstractArray, len::Int) 
  local sum::Float64 = 0.0;
  for i in 1:len; sum += abs2(a[i]);  end
  a[:] /= sqrt(sum)
end

# orthogonalize vector b to vector a
function orthogonalize(a::AbstractArray, b::AbstractArray, len::Int) 
  local dot::Complex{Float64} = 0.0;
  for i in 1:len; dot  += conj(a[i])*b[i]; end
  b[:] -= (dot*a[:])
end

#Main methods:

function applyGaugeFieldScaling!(gauge::Matrix{Complex{T}}, param::QJuliaInterface.QJuliaGaugeParam_qj) where T <: AbstractFloat

  vol    = param.X[1]*param.X[2]*param.X[3]*param.X[4] 
  volh   = Int(vol / 2) 
  volh_t = Int(param.X[1] / 2)*param.X[2]*param.X[3]*(param.X[4]-1)

  # Apply spatial scaling factor (u0) to spatial links
  gauge[:, :] /= param.anisotropy;
    
  # only apply T-boundary at edge nodes (always true for the single device)
  local last_node_in_t = (QUDARoutines.commCoords_qj(3) == QUDARoutines.commDim_qj(3)-1) ? true : false

  # create time direction views:
  even_tlinks = view(gauge, (9volh_t+1):9volh, 4)
  odd_tlinks  = view(gauge, 9(volh+volh_t)+1:9vol, 4) 

  # Apply boundary conditions to temporal links
  if param.t_boundary == QJuliaEnums.QJULIA_ANTI_PERIODIC_T && last_node_in_t
    println("Applying antiperiodic BC.")
    even_tlinks[:] *= -1.0
    odd_tlinks[:]  *= -1.0
  end

    
  if param.gauge_fix == QJuliaEnums.QJULIA_GAUGE_FIXED_YES
    println("Applying gauge fixing.")
    # set all gauge links (except for the last X[1]*X[2]*X[3]/2) to the identity,
    # to simulate fixing to the temporal gauge.
    local iMax = last_node_in_t ? volh_t : volh

    even_gauge = view(gauge, 1:9iMax, 4)
    odd_gauge  = view(gauge, 9volh+1:9(volh+iMax), 4)

    for i in 0:(iMax-1)
      for m in 0:2
	for n in 0:2
	  even_gauge[9i + 3m + n + 1] = (m==n) ? 1.0 : 0.0;
	  odd_gauge[ 9i + 3m + n + 1] = (m==n) ? 1.0 : 0.0;
	end # for n
      end # for m
    end #for i

  end # param.gauge_fix

end #applyGaugeFieldScaling!


function constructUnitGaugeField!(gauge::Matrix{Complex{T}}, param::QJuliaInterface.QJuliaGaugeParam_qj) where T <: AbstractFloat
  vol  = param.X[1]*param.X[2]*param.X[3]*param.X[4] 
  volh = Int(vol / 2) 

  even_gauge = view(gauge, 1:9volh, :)
  odd_gauge  = view(gauge, 9volh+1:9vol, :)
   
  for d in 1:4 
    for i in 0:(volh-1)
      for m in 0:2
        for n in 0:2
          even_gauge[9i + 3m + n + 1, d] = (m==n) ? 1.0 : 0.0;
          odd_gauge[9i + 3m + n + 1 , d] = (m==n) ? 1.0 : 0.0;
        end # for n
      end # for m
    end # for i
  end # for d
    
  applyGaugeFieldScaling!(gauge, param)

end # constructUnitGaugeField!


function constructGaugeField!(gauge::Matrix{Complex{T}}, param::QJuliaInterface.QJuliaGaugeParam_qj) where T <: AbstractFloat

  vol  = param.X[1]*param.X[2]*param.X[3]*param.X[4] 
  volh = Int(vol / 2) 

  println("Gauge field volume: ",  vol)

  even_gauge = view(gauge, 1:9volh, :)
  odd_gauge  = view(gauge, 9volh+1:9vol, :)

  #println(typeof(even_gauge)) 
  #println(typeof(odd_gauge) ) 
  #we do always need offset = 1
  for d in 1:4 
    for i in 0:(volh-1)
      for m in 0:2
        for n in 0:2
          even_gauge[9i + 3m + n + 1, d] = Complex{T}(rand(), rand())
          odd_gauge[9i + 3m + n + 1 , d] = Complex{T}(rand(), rand())
        end
      end
      #create a view for a given link
      local c = 1;
      gauge_link_col0 = view(even_gauge, (9i       +1):(9i+3c    ), d)
      gauge_link_col1 = view(even_gauge, (9i+3c    +1):(9i+3(c+1)), d)
      gauge_link_col2 = view(even_gauge, (9i+3(c+1)+1):(9i+3(c+2)), d)

      normalize(gauge_link_col1, 3)
      orthogonalize(gauge_link_col1, gauge_link_col2, 3)
      normalize(gauge_link_col2, 3)

      for i in 1:3; gauge_link_col0[i] = 0.0; end

      gauge_link_col0[1] = accumulateConjugateProduct(gauge_link_col0[1], gauge_link_col1[2], gauge_link_col2[3], +1.0)
      gauge_link_col0[1] = accumulateConjugateProduct(gauge_link_col0[1], gauge_link_col1[3], gauge_link_col2[2], -1.0)
      gauge_link_col0[2] = accumulateConjugateProduct(gauge_link_col0[2], gauge_link_col1[3], gauge_link_col2[1], +1.0)
      gauge_link_col0[2] = accumulateConjugateProduct(gauge_link_col0[2], gauge_link_col1[1], gauge_link_col2[3], -1.0)
      gauge_link_col0[3] = accumulateConjugateProduct(gauge_link_col0[3], gauge_link_col1[1], gauge_link_col2[2], +1.0)
      gauge_link_col0[3] = accumulateConjugateProduct(gauge_link_col0[3], gauge_link_col1[2], gauge_link_col2[1], -1.0)

      gauge_link_col0 = view(odd_gauge, (9i       +1):(9i+3c    ), d)
      gauge_link_col1 = view(odd_gauge, (9i+3c+1):(9i+3(c+1)), d)
      gauge_link_col2 = view(odd_gauge, (9i+3(c+1)+1):(9i+3(c+2)), d)

      normalize(gauge_link_col1, 3)
      orthogonalize(gauge_link_col1, gauge_link_col2, 3)
      normalize(gauge_link_col2, 3)

      for i in 1:3; gauge_link_col0[i] = 0.0; end

      gauge_link_col0[1] = accumulateConjugateProduct(gauge_link_col0[1], gauge_link_col1[2], gauge_link_col2[3], +1.0)
      gauge_link_col0[1] = accumulateConjugateProduct(gauge_link_col0[1], gauge_link_col1[3], gauge_link_col2[2], -1.0)
      gauge_link_col0[2] = accumulateConjugateProduct(gauge_link_col0[2], gauge_link_col1[3], gauge_link_col2[1], +1.0)
      gauge_link_col0[2] = accumulateConjugateProduct(gauge_link_col0[2], gauge_link_col1[1], gauge_link_col2[3], -1.0)
      gauge_link_col0[3] = accumulateConjugateProduct(gauge_link_col0[3], gauge_link_col1[1], gauge_link_col2[2], +1.0)
      gauge_link_col0[3] = accumulateConjugateProduct(gauge_link_col0[3], gauge_link_col1[2], gauge_link_col2[1], -1.0)
      
    end # for i
  end # for d

  if param.gtype == QJuliaEnums.QJULIA_WILSON_LINKS
    println("Applying scaling/BC on the gauge links")
    applyGaugeFieldScaling!(gauge, param)
  end

  if debug_constructGaugeField == true
    println("::> Begin debug info for function constructGaugeField...")
    dir = 1
    for i in 1:16
      println("Cehck value for index ", i, ",dir " , dir, " complex value is = ", gauge[i, dir])
    end
    println("<:: End debug info.")
  end

end #constructGaugeField!

function construct_gauge_field!(gauge::Matrix{Complex{T}}, gtype, param::QJuliaInterface.QJuliaGaugeParam_qj) where T <: AbstractFloat

  println("Working with gauge field type:",  typeof(gauge), " : ", length(gauge))

  if gtype == 0 
    println("Construct unit gauge field")
    constructUnitGaugeField!(gauge, param)
  elseif gtype == 1
    println("Construct random gauge field")
    constructGaugeField!(gauge, param)
  else
    println("Apply scaling...")
    applyGaugeFieldScaling!(gauge, param)
  end
end #construct_gauge_field!

using QJuliaFields

function construct_gauge_field!(field::QJuliaFields.QJuliaGenericField_qj, gtype, param::QJuliaInterface.QJuliaGaugeParam_qj)

  if(field.field_desc.geom != QJuliaEnums.QJULIA_VECTOR_GEOMETRY); error("Cannot apply on fields with non-vector geometry.");end
  if(field.field_desc.register_type != ComplexF64 && field.field_desc.register_type != ComplexF32 )
    error("Register type ", field.field_desc.register_type, " is currently not supported.")
  end

  construct_gauge_field!(field.v, gtype, param)

end

end #QJuliaGaugeUtils


